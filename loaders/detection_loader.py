import torch
import numpy as np
import os
from PIL import Image
from PIL import ImageFile
from torch.utils.data import DataLoader
from loaders import collate_detection_samples
ImageFile.LOAD_TRUNCATED_IMAGES = True


class DetectionLoader(torch.utils.data.Dataset):
    '''
    Detection loader loads samples, filters, and transforms 
    to the right dimensions, augmentation, etc.
    '''

    def __init__(self, data_path, area_filter=[0, 1000**2], categories_filter=None,
                 greyscale=False):
        super(DetectionLoader, self).__init__()
        self.greyscale = greyscale
        self.categories = {}
        self.transforms = None
        self.data_path = data_path
        self.area_limits = area_filter
        self.category_filter = {
            c: True for c in categories_filter
        } if categories_filter is not None else {}
        self.examples = []

    def __getitem__(self, idx):
        example = self.examples[idx]
        file_name = example['file_name']
        img = Image.open(os.path.join(self.data_path,
                                      file_name))

        try:
            if(self.greyscale):
                img = img.convert('L')
            else:
                img = img.convert('RGB')
        except IOError:
            print("Error: could not load {}".format(file_name))
            return

        width, height = example['width'], example['height']

        boxes = []
        labels = []

        for (box, label) in range(len(example['annotations'])):            
            if(label in self.categories):
                if(area(box) > self.area_limits[0] and area(box) < self.area_limits[1]):
                    boxes.append(box)
                    labels.append(self.categories[label]["remap_id"])

        sample = {
            'image': img,
            'boxes': torch.tensor(boxes),
            'labels': self.one_hot_matrix[labels],
            'labels_idx': torch.tensor(labels, dtype=torch.long),
            'width': width,
            'height': height,
            'id': example.Id()
        }

        if(self.transforms):
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.examples)
