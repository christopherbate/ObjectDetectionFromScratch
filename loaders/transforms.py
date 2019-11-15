import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
import math
import random
import matplotlib.pyplot as plt
import utils.boxes


class RandomResizeCropSample(object):
    def __init__(self, output_size, scale=(0.3, 1.0), ratio=(3.0/4.0, 4.0/3.0)):
        self.size = output_size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = Image.BILINEAR
        self.box_dtype = torch.float32

    @staticmethod
    def get_params(img, scale, ratio):
        area = img.size[0] * img.size[1]

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if (in_ratio < min(ratio)):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    @staticmethod
    def adjust_box(box, i, j, h, w, scales):
        box[0] = max(box[0] - j, 0)*scales[0]
        box[1] = (box[1] - i)*scales[1]
        box[2] = (box[2] - j)*scales[0]
        box[3] = (box[3] - i)*scales[1]
        return box

    def __call__(self, sample):
        '''
        img: PIL image
        '''

        new_sample = {}

        new_sample['boxes'] = torch.from_numpy(sample['boxes'])
        new_sample['labels'] = torch.from_numpy(sample['labels'])
        new_sample['labels_idx'] = sample['labels_idx']

        # i, j, h, w = self.get_params(sample['image'], self.scale, self.ratio)
        i, j, h, w = 0, 0, sample['image'].height, sample['image'].width

        # Crop the image
        new_sample['image'] = F.resized_crop(sample['image'],
                                             i, j, h, w,
                                             self.size,
                                             self.interpolation)
        new_sample['image'] = F.to_tensor(new_sample['image'])

        width_ratio = self.size[1] / w
        height_ratio = self.size[0] / h

        # Adjust all the boxes.
        new_sample["boxes"], inds = utils.boxes.adjust_boxes(
            new_sample["boxes"],
            (j, i, j+w, i+h),
            width_ratio, height_ratio)
        new_sample["boxes"] = new_sample["boxes"].to(self.box_dtype)

        new_sample['labels'] = new_sample['labels'][inds]

        return new_sample


class DetectionTransform(object):
    '''
    output_size : (h,w) tuple of the size the image should be resized to
    '''

    def __init__(self, output_size, greyscale=False, normalize=False):
        self.output_size = output_size
        self.resize_crop = RandomResizeCropSample(self.output_size)
        self.normalize = None
        if(normalize):
            mean = [0.485, 0.456, 0.406] if not greyscale else [0.456]
            std = [0.229, 0.224, 0.225] if not greyscale else [0.225]
            self.normalize = torchvision.transforms.Normalize(
                mean=mean, std=std)

    def __call__(self, sample):
        if(self.resize_crop):
            sample = self.resize_crop(sample)
        if(self.normalize):
            sample["image"] = self.normalize(sample["image"])

        return sample
