import unittest
import torch
from loaders.detection_loader import DetectionLoader
from torch.utils.tensorboard import SummaryWriter
from loaders.transforms import DetectionTransform
from loaders.detection_batch import collate_detection_samples, ObjectDetectionBatch
import time
from utils import normalize_tensor
writer = SummaryWriter(comment="loader-test")


class TestDetectionLoader(unittest.TestCase):
    def test_loader(self):
        transforms = DetectionTransform(output_size=(50, 100),
                                        greyscale=True, normalize=True)
        loader = DetectionLoader(
            "/home/chris/coco_tools/test.db", "/home/chris/datasets/coco/train2017/",
            area_filter=[100**2, 250**2],
            greyscale=True, transforms=transforms)

        for idx, sample in enumerate(loader):
            writer.add_image_with_boxes(
                "sample", normalize_tensor(sample["image"]), sample["boxes"], global_step=idx)
            writer.close()
            if idx == 100:
                break

    def test_batch(self):
        transforms = DetectionTransform(output_size=(50, 100),
                                        greyscale=True, normalize=True)
        ds = DetectionLoader(
            "/home/chris/coco_tools/test.db", "/home/chris/datasets/coco/train2017/",
            area_filter=[100**2, 250**2],
            greyscale=True, transforms=transforms)    

        loader = torch.utils.data.DataLoader(
            collate_fn=collate_detection_samples,
            dataset=ds,
            batch_size=10,
            pin_memory=True,
            num_workers=4)

        sample : ObjectDetectionBatch
        for idx, sample in enumerate(loader):
            writer.add_image_with_boxes(
                "sample", normalize_tensor(sample.images[0]), sample.boxes[0], global_step=idx)
            writer.close()
            if idx == 10:
                break


if __name__ == '__main__':
    unittest.main()
