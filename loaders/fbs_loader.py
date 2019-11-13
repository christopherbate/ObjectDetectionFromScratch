import flatbuffers
from loaders.dataset_fbs import Annotation
from loaders.dataset_fbs import Dataset
from loaders.dataset_fbs import Example
from loaders.dataset_fbs import Category
from loaders.dataset_fbs import BoundingBox

import numpy as np
import os
import time

import torch

from PIL import Image

import matplotlib.pyplot as plt


class FBSDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, database_path="/home/chirs/datasets/coco/train.db",
                 data_path="/home/chris/datasets/coco/train2017/",
                 greyscale=False,
                 transforms=None,
                 categories_filter: dict = None,
                 area_filter=[0, 800**2]):
        super(FBSDetectionDataset, self).__init__()
        self.database_path = database_path
        self.greyscale = greyscale
        self.categories = {}
        self.transforms = transforms
        self.data_path = data_path
        self.area_limits = area_filter
        with open(database_path, "rb") as infile:
            buf = infile.read()
        self.dataset = Dataset.Dataset.GetRootAsDataset(buf, 0)
        self.category_filter = {
            c: True for c in categories_filter} if categories_filter else None
        self.examples = []

        # Create the categories map
        for i in range(self.dataset.CategoriesLength()):
            cat = self.dataset.Categories(i)
            self.categories[cat.Id()] = {
                "name": cat.Name().decode('utf-8'),
                "remap_id": i,
                "num_images": cat.NumImages(),
                "example_list": cat.ExamplesAsNumpy(),
            }

        if(categories_filter is None):
            for i in range(self.dataset.ExamplesLength()):
                example = self.dataset.Examples(i)
                for j in range(example.AnnotationsLength()):
                    if(example.Annotations(j).Area() > self.area_limits[0]
                            and example.Annotations(j).Area() < self.area_limits[1]):
                        self.examples.append(example)
                        break
        else:
            self.allowed_ex_idxs = []
            for catId, cat in self.categories.items():
                if cat["name"] in self.category_filter:
                    self.allowed_ex_idxs += cat["example_list"].tolist()
            self.allowed_ex_idxs = np.unique(np.array(
                self.allowed_ex_idxs)).tolist()
            tmp_examples = [self.dataset.Examples(
                idx) for idx in self.allowed_ex_idxs]
            for example in tmp_examples:
                for j in range(example.AnnotationsLength()):
                    if(example.Annotations(j).Area() > self.area_limits[0]
                            and example.Annotations(j).Area() < self.area_limits[1]):
                        if(self.categories[example.Annotations(j).CatId()]["name"]
                                in self.category_filter):
                            self.examples.append(example)
                            break

            print("Based on filters, selected {} images".format(
                len(self.examples)))

        self.one_hot_matrix = np.eye(80, dtype=np.float32)
        self.total_cats = len(self.categories.keys())

    def print_categories(self):
        print("Categories: (total: {})".format(len(self.categories.keys())))
        for label_id, cat in self.categories.items():
            print("{} : {} : {} : {}".format(
                label_id, cat["remap_id"], cat["name"], cat["num_images"]))

    @staticmethod
    def collate_fn(example_list):
        assert(type(example_list) == list)

        # widths = [ex["image"].shape[-1] for ex in example_list]
        # heights = [ex["image"].shape[-2] for ex in example_list]
        imgs = [ex["image"] for ex in example_list]
        boxes = [ex["boxes"] for ex in example_list]
        labels = [ex["labels"] for ex in example_list]

        out = None
        if(torch.utils.data.get_worker_info() is not None):
            # This is from pytorch's default colate fn:
            # https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel_img = sum([x.numel() for x in imgs])
            storage = imgs[0].storage()._new_shared(numel_img)
            out = imgs[0].new(storage)

        # Boxes and labels use zero padding out to max length.
        # We use a util function used for RNNs, but the purpose/effect is the same.
        boxes = torch.nn.utils.rnn.pad_sequence(boxes, batch_first=True)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
        imgs = torch.stack(imgs, dim=0, out=out)

        batched_sample = {
            "image": imgs,
            "boxes": boxes,
            "labels": labels,
        }
        return batched_sample

    def __getitem__(self, idx):
        example = self.examples[idx]
        file_name = example.FileName().decode('utf-8')
        img = Image.open(os.path.join(self.data_path,
                                      file_name))
        if(self.greyscale):
            img = img.convert('L')
        else:
            img = img.convert('RGB')

        width, height = example.Width(), example.Height()

        boxes = []
        labels = []

        for i in range(example.AnnotationsLength()):
            ann = example.Annotations(i)
            box = ann.Bbox()
            category = self.categories[ann.CatId()]
            if(self.category_filter is not None):
                if(category["name"] in self.category_filter):
                    boxes.append([box.X1(), box.Y1(), box.X2(), box.Y2()])
                    labels.append(self.categories[ann.CatId()]["remap_id"])
            else:
                boxes.append([box.X1(), box.Y1(), box.X2(), box.Y2()])
                labels.append(self.categories[ann.CatId()]["remap_id"])

        sample = {
            'image': img,
            'boxes': np.array(boxes),
            'labels': self.one_hot_matrix[labels],
            'width': width,
            'height': height
        }

        if(self.transforms):
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.examples)
