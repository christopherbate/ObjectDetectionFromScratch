from torchvision.models.detection import rpn
import torch
from utils import boxes as boxops
from models.class_head import BoxPrediction
from models.backbone import Backbone
from models.anchor_generator import AnchorGenerator
import time
import numpy as np


class ObjectDetection(torch.nn.Module):
    '''
    predict_features is 'detail' or 'course'
    '''

    def __init__(self,
                 pos_threshold=0.4,
                 neg_threshold=0.1,
                 activation=torch.nn.functional.relu,
                 class_bias=-2.1,
                 num_classes=80,
                 predict_conf_threshold=0.75,
                 **kwargs):

        super(ObjectDetection, self).__init__(**kwargs)
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.predict_conf_threshold = predict_conf_threshold
        self.num_classes = num_classes

        self.IMAGE_SHAPE = (64, 128)

        # These are in order of low pass (gauss-smoothed wavelet) and
        # and non-smoothed wavelet
        self.FEATURE_COUNTS = (64,)
        self.ANCHOR_SIZES = ((16,32,64),)

        # These ratios are for all anchors
        self.ANCHOR_RATIOS = (1.0,0.5,2.0)

        self.backbone = Backbone()
        self.anchor_generator = AnchorGenerator(sizes=self.ANCHOR_SIZES,
                                                aspect_ratios=self.ANCHOR_RATIOS)
        self.box_prediction = BoxPrediction(num_features=self.FEATURE_COUNTS,
                                            num_class=num_classes,
                                            num_anchors=[len(anchors)*len(self.ANCHOR_RATIOS) for anchors in self.ANCHOR_SIZES])
        self.loss = torch.nn.BCEWithLogitsLoss(reduce=False)

        self.debug = True

        self.metrics = np.zeros(3)
        self.metric_count = 0

    def read_metrics(self):
        metrics = self.metrics / self.metric_count
        self.metric_count = 0
        self.metrics = np.zeros_like(self.metrics)
        return metrics

    def forward(self, sample):
        # Apply the backbone.
        # Results in a set of feature maps.
        img = sample['image']
        out = self.backbone(img)
        feature_maps = [out]

        # Make predictions
        box_data = self.box_prediction(feature_maps)
        logits = box_data['logits']
        probs = torch.sigmoid(logits.detach())

        # Create / retrieve cached anchors
        anchors = self.anchor_generator(sample['image'], out)

        fgbg_mask = torch.zeros(
            (logits.shape[0], logits.shape[1]), device=logits.device)
        class_targets = torch.zeros_like(logits)
     
        for idx, box_set in enumerate(sample["boxes"]):
            iou = boxops.anchor_box_iou(anchors, box_set)              
            fgbg_mask[idx], pos_ind, assignments = boxops.create_label_matrix(iou,
                                                                              pos_threshold=self.pos_threshold,
                                                                              neg_threshold=self.neg_threshold)                                                                     
            class_targets[idx, pos_ind] = sample['labels'][idx, assignments]
            

        '''
        Calculate focal loss
        '''
        class_loss = self.loss(logits, class_targets)
        focal_weights = torch.pow(class_targets-probs, 2)
        class_loss = torch.sum(class_loss*focal_weights, axis=2)

        '''
        Calcualte the predictions we should return for inference/visualization
        '''
        pos_labeled_mask = fgbg_mask == 1
        neg_labeled_mask = fgbg_mask == -1
        pos_predicted_conf, pos_predicted_targets = torch.max(probs, dim=2)
        pos_predicted_mask = pos_predicted_conf > self.pos_threshold

        '''
        Reduce the loss accross dimensions
        '''
        class_loss_reduced = class_loss[fgbg_mask != 0]
        class_loss_reduced = class_loss_reduced.sum()/(pos_labeled_mask.sum())

        '''
        in-model debugging
        '''
        if sample["debug"] == True:
            print("Class Loss: Red {:.2f} Mean {:.2f} Shape: {}".format(
                class_loss_reduced, class_loss.mean(), class_loss.shape))
            print("Class Loss Pos {:.2f} Neg {:.2f}".format(
                class_loss[pos_labeled_mask].mean(
                ), class_loss[neg_labeled_mask].mean()
            ))

            print("Mask shape: {}".format(fgbg_mask.shape))
            print("Class Probabilities {}".format(probs.shape))
            print("Class Target Shape: {}".format(class_targets.shape))
            print("Pos Target Mean: {:.2f}".format(
                class_targets[pos_labeled_mask, ...].sum(dim=-1).mean()))

            print("Pos Anchors {} Neg Anchors {} Neutral Anchors {}".format(
                torch.sum(fgbg_mask == 1), torch.sum(fgbg_mask == -1),
                torch.sum(fgbg_mask == 0)
            ))
            print("Focal Weights Shape {} Pos Mean {:.3f} Neg Mean {:.3f}".format(focal_weights.shape,
                                                                                  focal_weights[pos_labeled_mask].mean(
                                                                                  ),
                                                                                  focal_weights[neg_labeled_mask].mean()))
            val, _ = probs[pos_labeled_mask, ...].max(dim=-1)
            print("Avg PosAnchor Confidence {}   {}".format(
                val.mean(), val.shape))
            print("\n")

        losses = {
            'class_loss': class_loss_reduced
        }
        data = {
            'pos_predicted_anchors': [anchors[mask] for mask in pos_predicted_mask],
            'pos_predicted_confidence': [pos_predicted_conf[idx, mask] for idx, mask in enumerate(pos_predicted_mask)],
            'pos_predicted_labels': pos_predicted_targets,
            'pos_labeled_anchors': [anchors[mask] for mask in pos_labeled_mask],
            'pos_labeled_confidence': probs[pos_labeled_mask],
        }

        return losses, data
