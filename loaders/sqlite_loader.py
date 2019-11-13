import sqlite3

import numpy as np
import os
import time

import torch
import torchvision
from PIL import Image

import matplotlib.pyplot as plt

def load_img(base_path: str,
             filename: str,
             greyscale: bool = False):
    '''
    Loads image from file and returns as PIL image
    '''

    img = Image.open(os.path.join(base_path,
                                  filename))
    if(greyscale):
        img = img.convert('L')
    else:
        img = img.convert('RGB')

    return img

class SqlClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, sqlite_file: str, 
                base_path: str = "/home/chris/datasets/coco/train2017",
                 transforms=None, greyscale=True):
        self.sqlite_file = sqlite_file
        self.examples = []
        self.greyscale = greyscale
        self.transforms = ClassificationTransform(
            greyscale=self.greyscale, normalize=True)
        self.base_path = base_path
        self.categories = {}
        self.class_strings = []
        self.cat_distribution = []

        with sqlite3.connect(self.sqlite_file) as conn:
            for row in conn.execute("select filename, category, area from thumbnails where area > 5000"):
                self.examples.append(row)
            for row in conn.execute("select id, name, cocoid from categories order by cocoid"):
                self.categories[row[2]] = {
                    "name": row[1].decode('utf-8'),
                    "id": row[0]
                }
                self.class_strings.append(row[1].decode('utf-8'))

            for row in conn.execute("select id, count(id) from thumbnails group by category order by 1"):
                self.cat_distribution.append(row[1])        

        self.one_hot_matrix = torch.eye(len(list(self.categories.keys())))

    def __getitem__(self, idx):
        example = self.examples[idx]
        img = load_img(self.base_path, example[0], self.greyscale)

        label = example[1]
        label_idx = self.categories[label]["id"]

        sample = {
            'image': img,
            'label': self.categories[label]["id"],
            'one_hot_label': self.one_hot_matrix[label_idx],
            'area': example[2]
        }

        if(self.transforms):
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.examples)

    @staticmethod
    def collate_fn(example_list):
        assert(type(example_list) == list)

        cats = torch.tensor([ex["label"] for ex in example_list])
        imgs = [ex["image"] for ex in example_list]
        oh_labels =  torch.stack([ex["one_hot_label"] for ex in example_list], dim=0)

        out = None
        if(torch.utils.data.get_worker_info() is not None):
            # This is from pytorch's default colate fn:
            # https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel_img = sum([x.numel() for x in imgs])
            storage = imgs[0].storage()._new_shared(numel_img)
            out = imgs[0].new(storage)

        imgs = torch.stack(imgs, dim=0, out=out)

        batched_sample = {
            "image": imgs,
            "labels": cats,
            "one_hot_labels": oh_labels
        }
        return batched_sample


class ClassificationTransform(object):
    def __init__(self, greyscale=False, normalize=False):
        self.output_size = (256, 256)
        self.resize_crop = torchvision.transforms.Resize(self.output_size)
        self.normalize = None
        self.to_tensor = torchvision.transforms.ToTensor()
        if(normalize):
            mean = [0.485, 0.456, 0.406] if not greyscale else [0.456]
            std = [0.229, 0.224, 0.225] if not greyscale else [0.225]
            self.normalize = torchvision.transforms.Normalize(
                mean=mean, std=std)

    def __call__(self, sample):
        if(self.resize_crop):
            sample["image"] = self.resize_crop(sample["image"])
        sample["image"] = self.to_tensor(sample["image"])
        if(self.normalize):
            sample["image"] = self.normalize(sample["image"])

        return sample
