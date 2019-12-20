import torch
import numpy as np
import os
import sqlite3
from PIL import Image
from PIL import ImageFile
from torch.utils.data import DataLoader
import time
from loaders import collate_detection_samples
ImageFile.LOAD_TRUNCATED_IMAGES = True


class DetectionLoader(torch.utils.data.Dataset):
    '''
    Detection loader loads samples, filters, and transforms 
    to the right dimensions, augmentation, etc.
    '''

    def __init__(self, db_path, data_path, area_filter=[0, 1000**2], categories_filter=None,
                 greyscale=False, transforms=None):
        super(DetectionLoader, self).__init__()
        self.db_path = db_path
        self.greyscale = greyscale
        self.categories = {}
        self.transforms = transforms
        self.data_path = data_path
        self.area_limits = area_filter if area_filter is not None else [
            0, 1000**2]
        self.category_filter = {
            c: True for c in categories_filter
        } if categories_filter is not None else None
        self.examples = []
        self.class_strings = []

        startTime = time.time()
        print("Selecting examples from database.")
        with sqlite3.connect(self.db_path) as conn:
            for idx, row in enumerate(conn.execute("select id, name, coco_id from categories order by coco_id")):
                if self.category_filter is None or row[1] in self.category_filter:
                    self.categories[row[2]] = {
                        "name": row[1],
                        "id": idx,
                        "coco_id": row[2]
                    }
                    self.class_strings.append(row[1])

            print("Categories")
            allowed_categories = list(self.categories.keys())
            allowed_categories = "("+",".join([str(cid)
                                               for cid in allowed_categories])+")"
            print("Allowed categories: ", allowed_categories)
            for row in conn.execute("select filename, coco_id, width, height, area from images"):
                anns = conn.execute(
                    "select box, cat_coco_id from annotations where image_coco_id = (?) and is_crowd = 0 and area > ? and area < ? and cat_coco_id in {}".format(
                        allowed_categories),
                    (row[1], self.area_limits[0], self.area_limits[1])).fetchall()
                boxes = [[float(fStr) for fStr in a[0].split(',')]
                         for a in anns]
                labels = [self.categories[a[1]]["id"] for a in anns]
                if len(labels) == 0:
                    continue
                ex = {
                    "filename": row[0],
                    "coco_id": row[1],
                    "boxes": boxes,
                    "labels": labels,
                    "dims": [row[2], row[3], row[3]],
                    "width": row[2],
                    "height": row[3],
                    "area": row[4]
                }
                self.examples.append(ex)

        duration = time.time()-startTime
        print("Selected {} examples in {:.3f} seconds".format(
            len(self.examples), duration))
        self.one_hot_matrix = torch.eye(
            len(self.categories), dtype=torch.float32)

    def __getitem__(self, idx):
        example = self.examples[idx]
        file_name = example["filename"]
        img = Image.open(os.path.join(self.data_path,
                                      file_name))
        try:
            if(self.greyscale):
                img = img.convert('L')
            else:
                img = img.convert('RGB')
        except Exception as e:
            print("Error: could not load {}".format(file_name))
            print(e)
            return

        width, height = example['width'], example['height']
        oh = self.one_hot_matrix[example["labels"]]
        sample = {
            'image': img,
            'boxes': torch.tensor(example["boxes"], dtype=torch.float32),
            'labels': oh,
            'labels_idx': torch.tensor(example["labels"], dtype=torch.long),
            'width': width,
            'height': height,
            'id': example["coco_id"]
        }
        if(self.transforms):
            sample = self.transforms(sample)
        return sample

    def __len__(self):
        return len(self.examples)
