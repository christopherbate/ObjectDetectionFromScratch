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

def train_model(args):
    model = Backbone()
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

    loader = torch.utils.data.DataLoader(
        collate_fn=FBSDetectionDataset.collate_fn,
        dataset=dataset, batch_size=args.batch_size)

    start_time = time.time()
    for epoch in range(1, args.epochs+1):
        for idx, batch in enumerate(loader):
            # print(batch['image'].shape)

            if idx % args.log_interval == 0:
                step = (epoch-1)*len(loader)+idx+1

                print("Step {} Batch {}/{} Loss : {}".format(
                    step, idx, len(loader), 0
                ))

                # grid = torchvision.utils.make_grid(
                #     batch["image"], normalize=True)
                # plt.imshow(grid.permute(1, 2, 0))
                # plt.show()
                
                writer.add_image_with_boxes("training_images", normalize_tensor(batch["image"][0]),
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

    args = parser.parse_args()
    print(args)
    train_model(args)
