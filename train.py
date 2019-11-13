import torch
import time
import argparse
from loaders.fbs_loader import FBSDetectionDataset
from loaders.transforms import DetectionTransform
from models.backbone import Backbone
import torchvision
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from utils.image import normalize_tensor
from models.detection_model import ObjectDetection
from torchvision.ops import nms


def train_model(args):
    writer = SummaryWriter()

    transforms = DetectionTransform(greyscale=True, normalize=True)
    dataset = FBSDetectionDataset(
        database_path=args.db,
        data_path=args.images,
        greyscale=True,
        transforms=transforms,
        categories_filter={
            'person': True
        },
        area_filter=[100**2, 500**2]
    )

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
        collate_fn=FBSDetectionDataset.collate_fn,
        dataset=datasets[0], batch_size=args.batch_size)

    validation_loader = torch.utils.data.DataLoader(
        dataset=datasets[1],
        batch_size=args.batch_size,
        collate_fn=FBSDetectionDataset.collate_fn,
    )

    '''
    Select device (cpu/gpu)
    '''
    device = torch.device(args.device)

    '''
    Create the model and transfer weights to device
    '''
    model = ObjectDetection(
        pos_threshold=0.5,
        neg_threshold=0.1,
        class_bias=-1.1,
        num_classes=80,
        predict_conf_threshold=0.5
    ).to(device)

    '''
    Select optimizer
    '''
    optim = torch.optim.SGD(params=model.parameters(),
                            lr=args.lr, momentum=0.5)

    '''
    Outer training loop
    '''
    start_time = time.time()
    for epoch in range(1, args.epochs+1):
        '''
        Inner training loop
        '''
        print("\n BEGINNING TRAINING STEP EPOCH {}".format(epoch))
        for idx, batch in enumerate(loader):
            '''
            Reset gradient
            '''
            optim.zero_grad()

            '''
            Push the data to the gpu (if necessary)
            '''
            batch['image'] = batch['image'].to(device)
            batch['labels'] = batch['labels'].to(device)
            batch['boxes'] = batch['boxes'].to(device)
            batch['debug'] = True if idx % args.log_interval == 0 else False

            '''
            Run the model
            '''
            losses, model_data = model(batch)

            '''
            Calc gradient and step optimizer.
            '''
            losses['class_loss'].backward()
            optim.step()

            if idx % args.log_interval == 0:
                step = (epoch-1)*len(loader)+idx+1

                print("Step {} Batch {}/{} Loss : {}".format(
                    step, idx, len(loader), 0
                ))

                '''
                Option1 : Plot with PYPLOT
                '''
                # grid = torchvision.utils.make_grid(
                #     batch["image"], normalize=True)
                # plt.imshow(grid.permute(1, 2, 0))
                # plt.show()

                '''
                    Option2 : Plot with tensorboard
                '''
                writer.add_image_with_boxes("training_images", normalize_tensor(batch["image"][0]),
                                            box_tensor=batch["boxes"][0], global_step=step)
                writer.add_scalar(
                    "batch_time", (time.time()-start_time)*1000.0, global_step=step)
                writer.add_scalar(
                    "training_loss", losses['class_loss'].item(),
                    global_step=step
                )
                writer.add_image_with_boxes("training_img_predicted_anchors", normalize_tensor(batch["image"])[0],
                                            model_data["pos_predicted_anchors"][0], global_step=step)

                '''
                Apply nms to predictions.
                '''
                keep_ind = nms(model_data["pos_predicted_anchors"][0],
                               model_data["pos_predicted_confidence"][0], iou_threshold=0.5)
                print("Indicies after NMS: ", keep_ind, model_data["pos_predicted_confidence"][0].shape, model_data["pos_predicted_anchors"][0].shape)
                writer.add_image_with_boxes("training_img_predicted_post_nms", normalize_tensor(batch["image"])[0],
                                            model_data["pos_predicted_anchors"][0][keep_ind], global_step=step)

                writer.close()

        '''
        Inner validation loop
        '''
        print("\nBEGINNING VALIDATION STEP {}\n".format(epoch))
        with torch.no_grad():
            for idx, batch in enumerate(validation_loader):
                if idx % args.log_interval == 0:
                    step = (epoch-1)*len(loader)+idx+1

                    print("Step {} Batch {}/{} Loss : {}".format(
                        step, idx, len(loader), 0
                    ))

                    writer.add_image_with_boxes("validation_images", normalize_tensor(batch["image"][0]),
                                                box_tensor=batch["boxes"][0], global_step=step)
                    writer.add_scalar(
                        "batch_time", (time.time()-start_time)*1000.0, global_step=step)

                    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str)
    parser.add_argument("--images", type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--lr", type=float, default=0.01)

    args = parser.parse_args()
    print(args)
    train_model(args)
