import torch
import time
import argparse
from torch.utils.tensorboard import SummaryWriter
from torchvision.ops import nms
import torchvision

from loaders import FBSDetectionDataset
from loaders import ObjectDetectionBatch, collate_detection_samples
from loaders import DetectionTransform
from models.detection_model import ObjectDetection
from utils.image import normalize_tensor


def train_model(args):
    writer = SummaryWriter()

    transforms = DetectionTransform(output_size=args.resize,
                                    greyscale=True, normalize=True)

    dataset = FBSDetectionDataset(
        database_path=args.db,
        data_path=args.images,
        greyscale=True,
        transforms=transforms,
        categories_filter=["person"],
        area_filter=[50**2, 250**2]
    )
    dataset.print_categories()

    ''' 
    Split dataset into train and validation
    '''
    train_len = int(0.75*len(dataset))
    dataset_lens = [train_len, len(dataset)-train_len]
    print("Splitting dataset into pieces: ", dataset_lens)
    datasets = torch.utils.data.random_split(dataset, dataset_lens)
    print(datasets)

    '''
    Setup the data loader objects (collation, batching)
    '''
    loader = torch.utils.data.DataLoader(
        collate_fn=collate_detection_samples,
        dataset=datasets[0],
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_data_workers)

    validation_loader = torch.utils.data.DataLoader(
        dataset=datasets[1],
        batch_size=args.batch_size,
        pin_memory=True,
        collate_fn=collate_detection_samples,
        num_workers=args.num_data_workers
    )

    '''
    Select device (cpu/gpu)
    '''
    device = torch.device(args.device)

    '''
    Create the model and transfer weights to device
    '''
    model = ObjectDetection(
        input_image_shape=args.resize,
        pos_threshold=args.pos_anchor_iou,
        neg_threshold=args.neg_anchor_iou,
        num_classes=len(dataset.categories),
        predict_conf_threshold=args.filter_conf
    ).to(device)

    '''
    Select optimizer
    '''
    optim = torch.optim.SGD(params=model.parameters(),
                            lr=args.lr, momentum=0.9)

    def lr_schedule(epoch):
        if(epoch >= 60):
            return 0.01
        if epoch >= 20:
            return 0.1
        if epoch >= 5:
            return 1
        if epoch >= 1:
            return 0.1
        return 0.1

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=[lr_schedule])

    '''
    Outer training loop
    '''
    for epoch in range(1, args.epochs+1):
        '''
        Inner training loop
        '''
        print("\n BEGINNING TRAINING STEP EPOCH {} LR {}".format(
            epoch, lr_scheduler.get_lr()))
        cummulative_loss = 0.0
        start_time = time.time()

        batch: ObjectDetectionBatch
        model.train()
        for idx, batch in enumerate(loader):
            '''
            Reset gradient
            '''
            optim.zero_grad()

            '''
            Push the data to the gpu (if necessary)
            '''
            batch.to(device)
            batch.debug = False  # True if idx % args.log_interval == 0 else False

            '''
            Run the model
            '''
            losses, model_data = model(batch)
            cummulative_loss += losses["class_loss"].item()

            '''
            Calc gradient and step optimizer.
            '''
            losses['class_loss'].backward()
            optim.step()

            '''
            Log Metrics and Visualizations
            '''
            if (idx) % args.log_interval == 0:
                step = (epoch-1)*len(loader)+idx+1

                print("Ep {} Training Step {} Batch {}/{} Loss : {:.3f}".format(
                    epoch, step, idx, len(loader), cummulative_loss
                ))

                '''
                Save visualizations and metrics with tensorboard
                Note: For research, to reproduce graphs you will want some way to save the collected metrics (e.g. the loss values)
                to an array for recreating figures for a paper. To do so, metrics are often wrapped in a "metering" class
                that takes care of logging to tensorboard, resetting cumulative metrics, saving arrays, etc.
                '''

                '''
                training_image - the raw training images with box labels

                training_image_predicted_anchors - predictions for the same image, using basic thresholding (0.7 confidence on the logit)
                
                training_image_predicted_post_nms - predictions for the same image, filtered at 0.7 confidence followed by Non-Max-Suppression

                training_image_positive_anchors - shows anchors which received a positive label in the labeling step in the model
                '''
                sample_image = normalize_tensor(batch.images[0])
                writer.add_image_with_boxes("training_image", sample_image,
                                            box_tensor=batch.boxes[0], global_step=step)
                writer.add_image_with_boxes("training_image_predicted_anchors", sample_image,
                                            model_data["pos_predicted_anchors"][0], global_step=step)

                keep_ind = nms(model_data["pos_predicted_anchors"][0],
                               model_data["pos_predicted_confidence"][0], iou_threshold=args.nms_iou)

                writer.add_image_with_boxes("training_image_predicted_post_nms", sample_image,
                                            model_data["pos_predicted_anchors"][0][keep_ind], global_step=step)
                writer.add_image_with_boxes("training_image_positive_anchors", sample_image,
                                            box_tensor=model_data["pos_labeled_anchors"][0], global_step=step)

                '''
                First layer conv filters
                '''
                scale_each = True
                filter_grid = torchvision.utils.make_grid(
                    torch.cat([model.backbone.first_conv.weight.data[:, :3, :, :],
                               model.backbone.first_conv.weight.grad[:, :3, :, :]], dim=0),
                    scale_each=scale_each,
                    normalize=True)
                writer.add_image("conv1", filter_grid, global_step=step)

                filter_grid = torchvision.utils.make_grid(
                    torch.cat([model.backbone.res_blks[0].conv1.weight.data[:, :3, :, :],
                               model.backbone.res_blks[0].conv1.weight.grad[:, :3, :, :]], dim=0),
                    normalize=True, scale_each=scale_each)
                writer.add_image("conv2", filter_grid, global_step=step)

                filter_grid = torchvision.utils.make_grid(
                    torch.cat([model.backbone.res_blks[1].conv2.weight.data[:, :3, :, :],
                               model.backbone.res_blks[1].conv2.weight.grad[:, :3, :, :]], dim=0),
                    normalize=True, scale_each=scale_each)
                writer.add_image("conv5", filter_grid, global_step=step)

                filter_grid = torchvision.utils.make_grid(
                    torch.sum(model_data['feature_maps'][0]
                              [:10, :, :, :], dim=1, keepdim=True),
                    scale_each=scale_each, normalize=True)
                writer.add_image("conv1_features",
                                 filter_grid, global_step=step)

                '''
                Scalars - batch_time, training loss
                '''
                writer.add_scalar(
                    "batch_time", ((time.time()-start_time)/float(args.log_interval))*1000.0, global_step=step)
                writer.add_scalar(
                    "training_loss", losses['class_loss'].item(),
                    global_step=step
                )

                writer.add_scalar(
                    "avg_pos_labeled_anchor_conf", torch.tensor(
                        [c.mean() for c in model_data["pos_labeled_confidence"]]).mean().item(),
                    global_step=step
                )

                writer.add_scalar(
                    "conv1_grad_norm", model.backbone.first_conv.weight.grad.norm().item(),
                    global_step=step
                )
                writer.add_scalar(
                    "backbone_last_grad_norm", model.backbone.res_blks[-1].conv2.weight.grad.norm(
                    ).item(),
                    global_step=step
                )

                start_time = time.time()

                writer.close()

            '''
            Reset metric meters as necessary
            '''
            if idx % args.metric_interval == 0:
                cummulative_loss = 0.0

        '''
        Inner validation loop
        '''
        print("\nBEGINNING VALIDATION STEP {}\n".format(epoch))
        with torch.no_grad():
            batch: ObjectDetectionBatch
            model.eval()
            for idx, batch in enumerate(validation_loader):
                '''
                Push the data to the gpu (if necessary)
                '''
                batch.to(device)
                batch.debug = False  # True if idx % args.log_interval == 0 else False

                '''
                Run the model
                '''
                losses, model_data = model(batch)

                if idx % args.log_interval == 0:
                    step = (epoch-1)*len(validation_loader)+idx+1

                    print("Ep {} Validation Step {} Batch {}/{} Loss : {:.3f}".format(
                        epoch, step, idx, len(
                            validation_loader), losses["class_loss"].item()
                    ))

                    '''
                    Log Images
                    '''
                    sample_image = normalize_tensor(batch.images[0])
                    writer.add_image_with_boxes("validation_images", sample_image,
                                                box_tensor=batch.boxes[0], global_step=step)

                    writer.add_image_with_boxes("validation_img_predicted_anchors", sample_image,
                                                model_data["pos_predicted_anchors"][0], global_step=step)

                    keep_ind = nms(model_data["pos_predicted_anchors"][0],
                                   model_data["pos_predicted_confidence"][0], iou_threshold=0.5)

                    writer.add_image_with_boxes("validation_img_predicted_post_nms", sample_image,
                                                model_data["pos_predicted_anchors"][0][keep_ind], global_step=step)

                    '''
                    Log Scalars
                    '''
                    writer.add_scalar(
                        "validation_loss", losses['class_loss'].item(),
                        global_step=step
                    )
                    writer.close()

        lr_scheduler.step()
        print("Stepped learning rate. Rate is now: ", lr_scheduler.get_lr())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str)
    parser.add_argument("--images", type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--metric_interval", type=int, default=10)
    parser.add_argument('--resize', nargs='+', type=int, default=(128, 64))
    parser.add_argument('--neg_anchor_iou', type=float, default=0.4)
    parser.add_argument('--pos_anchor_iou', type=float, default=0.5)
    parser.add_argument('--nms_iou', type=float, default=0.4)
    parser.add_argument('--num_data_workers', type=int,
                        default=torch.get_num_threads())
    parser.add_argument('--filter_conf',  type=float, default=0.3)
    args = parser.parse_args()

    # Turn the resize parameter into the reverse (WH -> HW)
    args.resize = args.resize[::-1]

    print(args)
    train_model(args)
