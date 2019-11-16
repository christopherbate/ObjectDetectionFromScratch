from loaders.dataset_fbs import BoundingBox
from loaders.dataset_fbs import Category
from loaders.dataset_fbs import Example
from loaders.dataset_fbs import Dataset
from loaders.dataset_fbs import Annotation
import numpy as np
import os
import torch
from PIL import Image
from PIL import ImageFile
from torch.utils.data import DataLoader
from loaders import collate_detection_samples
ImageFile.LOAD_TRUNCATED_IMAGES = True


class FBSDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, database_path="/home/chirs/datasets/coco/train.db",
                 data_path="/home/chris/datasets/coco/train2017/",
                 greyscale=False,
                 transforms=None,
                 categories_filter: list = None,
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
        unique_idx = 0
        for i in range(self.dataset.CategoriesLength()):
            cat = self.dataset.Categories(i)

            if self.category_filter is not None:
                self.category_filter: dict
                if cat.Name().decode('utf-8') not in self.category_filter:
                    continue

            self.categories[cat.Id()] = {
                "name": cat.Name().decode('utf-8'),
                "remap_id": unique_idx,
                "num_images": cat.NumImages(),
                "example_list": cat.ExamplesAsNumpy(),
            }
            unique_idx += 1

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
                        if example.Annotations(j).CatId() in self.categories:
                            self.examples.append(example)
                            break

        print("Based on filters, selected {} images".format(
            len(self.examples)))
        print("Filtered Dataset has {} categories".format(len(self.categories)))

        self.one_hot_matrix = torch.eye(
            len(self.categories), dtype=torch.float32)

    def print_categories(self):
        print("Categories: (total: {})".format(len(self.categories.keys())))
        for label_id, cat in self.categories.items():
            print("{} : {} : {} : {}".format(
                label_id, cat["remap_id"], cat["name"], cat["num_images"]))

    def verify_images(self):
        loader = DataLoader(dataset=self, batch_size=100,
                            collate_fn=collate_detection_samples, num_workers=12)
        for idx, sample in enumerate(loader):
            if idx % 10 == 0:
                print(idx)
            continue


    def __getitem__(self, idx):
        example = self.examples[idx]
        file_name = example.FileName().decode('utf-8')
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

        width, height = example.Width(), example.Height()

        boxes = []
        labels = []

        for i in range(example.AnnotationsLength()):
            ann = example.Annotations(i)
            box = ann.Bbox()

            if(ann.CatId() in self.categories):
                boxes.append([box.X1(), box.Y1(), box.X2(), box.Y2()])
                labels.append(self.categories[ann.CatId()]["remap_id"])

        sample = {
            'image': img,
            'boxes': torch.tensor(boxes),
            'labels': self.one_hot_matrix[labels],
            'labels_idx': torch.tensor(labels, dtype=torch.long),
            'width': width,
            'height': height
        }

        if(self.transforms):
            sample = self.transforms(sample)
        return sample

    def __len__(self):
        return len(self.examples)
