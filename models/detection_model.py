import torch
import numpy as np
from loaders import ObjectDetectionBatch
from utils import boxes as boxops
from models.class_head import BoxPrediction
from models.backbone import Backbone
from models.anchor_generator import AnchorGenerator


class ObjectDetection(torch.nn.Module):
    '''
        input_image_shape : The size of the input image.
                            In our system, this is fixed. For non-fixed systems, it should be
                            the average or median size.

        num_classes : The number of classes that should be predicted

        predict_conf_threshold : All predictions are filtered by this amount. E.g. if the model
                                predicts an object is present by 0.8 confidence, then we keep that
                                as a positive prediction which could be passed to NMS. Otherwise,
                                if confidence is < 0.75, it is is not passed on.
    '''

    def __init__(self,
                 input_image_shape,
                 num_classes,
                 pos_threshold=0.5,
                 neg_threshold=0.1,
                 predict_conf_threshold=0.75,
                 **kwargs):

        super(ObjectDetection, self).__init__(**kwargs)
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.predict_conf_threshold = predict_conf_threshold
        self.num_classes = num_classes
        self.input_image_shape = input_image_shape

        # The feature counts / depth for each feature map considered
        # for the class regression head
        self.FEATURE_COUNTS = (128,)

        # Anchor sizes  (per layer)
        # The anchors sizes need to scale to cover the sizes of possible objects
        # in the dataset.
        # You can either set an absolute pixel value, or set based off size of image.
        width = input_image_shape[-1]
        self.ANCHOR_SIZES = ((
            width/6,
            width/5,
            width/4,
            width/3),)

        # These ratios are for all anchors
        self.ANCHOR_RATIOS = (1.0,)

        self.backbone = Backbone()
        self.anchor_generator = AnchorGenerator(sizes=self.ANCHOR_SIZES,
                                                aspect_ratios=self.ANCHOR_RATIOS)

        self.box_prediction = BoxPrediction(num_features=self.FEATURE_COUNTS,
                                            num_class=num_classes,
                                            batch_norm=True,
                                            last_bias=-9.0,
                                            num_anchors=[len(anchors)*len(self.ANCHOR_RATIOS) for anchors in self.ANCHOR_SIZES])

        self.loss = torch.nn.BCEWithLogitsLoss(reduce=False)

    def forward(self, batch: ObjectDetectionBatch):
        '''
        '''
        # Apply the backbone.
        # Results in a set of feature maps.
        out, bb_maps = self.backbone(batch.images)
        feature_maps = [out]

        # Make predictions
        logits = self.box_prediction(feature_maps)
        probs = torch.sigmoid(logits)
        # Create / retrieve cached anchors
        anchors = self.anchor_generator(batch.images, feature_maps)

        fgbg_mask = torch.zeros(
            (logits.shape[0], logits.shape[1]), device=logits.device)
        class_targets = torch.zeros_like(logits)

        positive_anchor_indices = []
        positive_anchor_class_assignments = []
        positive_anchor_confidences = []

        with torch.no_grad():
            for idx, box_set in enumerate(batch.boxes):
                iou = boxops.anchor_box_iou(anchors, box_set)
                fgbg_mask[idx], pos_ind, assignments = boxops.create_label_matrix(iou,
                                                                                  pos_threshold=self.pos_threshold,
                                                                                  neg_threshold=self.neg_threshold)
                positive_anchor_indices.append(pos_ind)
                positive_anchor_class_assignments.append(assignments)
                class_targets[idx, pos_ind] = batch.labels[idx, assignments]
                labels_idx = batch.labels_idx[idx, assignments]
                positive_anchor_confidences.append(
                    probs[idx, pos_ind, labels_idx])

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
        pos_predicted_mask = pos_predicted_conf > self.predict_conf_threshold

        '''
        Reduce the loss accross dimensions
        '''
        class_loss_reduced = class_loss[fgbg_mask != 0]
        class_loss_reduced = class_loss_reduced.sum()/(pos_labeled_mask.sum())

        '''
        in-model debugging
        '''
        if batch.debug == True:
            print("Class Loss: Red {:.4f} Mean {:.4f} Shape: {}".format(
                class_loss_reduced, class_loss.mean(), class_loss.shape))
            print("Class Loss Pos {:.4f} Neg {:.4f}".format(
                class_loss[pos_labeled_mask].mean(
                ), class_loss[neg_labeled_mask].mean()
            ))

            print("Mask shape: {}".format(fgbg_mask.shape))
            print("Class Probabilities {}".format(probs.shape))
            print("Class Target Shape: {}".format(class_targets.shape))

            # Uncomment to check that all positive anchors are receiving a target of "1.0"
            # print("Pos Target Check: {:.2f}".format(
            # class_targets[pos_labeled_mask, ...].sum(dim=-1).mean()))

            # Prints the number of positive, neural, and negatively labeled anchors.
            print("Pos Anchors {} Neg Anchors {} Neutral Anchors {}".format(
                torch.sum(fgbg_mask == 1), torch.sum(fgbg_mask == -1),
                torch.sum(fgbg_mask == 0)
            ))

            # Prints focal loss means
            print("Focal Weights Shape {} Pos Mean {:.3f} Neg Mean {:.3f}".format(focal_weights.shape,
                                                                                  focal_weights[pos_labeled_mask].mean(
                                                                                  ),
                                                                                  focal_weights[neg_labeled_mask].mean()))

            # Uncomment to check that all ground truth boxes receive some anchor
            # assignment (no NaN values in the printed tensor), and to monitor
            # avg confidence. (This is plotted in Tensorboard so should probably be kept commented here)
            # print("Avg PosAnchor Confidence {}".format(
            #     torch.tensor([c.mean() for c in positive_anchor_confidences])))
            print("\n")

        losses = {
            'class_loss': class_loss_reduced
        }

        data = {
            'pos_predicted_anchors': [anchors[mask] for mask in pos_predicted_mask],
            'pos_predicted_confidence': [pos_predicted_conf[idx, mask] for idx, mask in enumerate(pos_predicted_mask)],
            'pos_predicted_labels': pos_predicted_targets,
            'pos_labeled_anchors': [anchors[mask] for mask in pos_labeled_mask],
            'pos_labeled_confidence': positive_anchor_confidences,
            'feature_maps': bb_maps
        }

        return losses, data
