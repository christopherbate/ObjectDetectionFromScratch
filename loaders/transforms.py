import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
import math
import random
import matplotlib.pyplot as plt


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
        new_sample["boxes"], inds = adjust_boxes(
            new_sample["boxes"],
            (j, i, j+w, i+h),
            width_ratio, height_ratio)
        new_sample["boxes"] = new_sample["boxes"].to(self.box_dtype)

        new_sample['labels'] = new_sample['labels'][inds]

        return new_sample


class DetectionTransform(object):
    def __init__(self, greyscale=False, normalize=False):
        self.output_size = (64, 128)
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


def adjust_boxes(boxes, crop_coords,
                 scale_x=1.0, scale_y=1.0):
    '''
    Adjusts boxes after crop+rescale of image

    crop_coords: [x1, y1, x2, y2]

    returns new boxes and indices of boxes retained
    '''
    intersections = intersection(boxes,
                                 torch.tensor(crop_coords))
    width = crop_coords[2] - crop_coords[0]
    height = crop_coords[3] - crop_coords[1]
    boxes[:, 0] = boxes[:, 0] - crop_coords[0]
    boxes[:, 1] = boxes[:, 1] - crop_coords[1]
    boxes[:, 2] = boxes[:, 2] - crop_coords[0]
    boxes[:, 3] = boxes[:, 3] - crop_coords[1]

    boxes[:, 0] = torch.clamp(boxes[:, 0],
                              min=0.0, max=width)*scale_x
    boxes[:, 2] = torch.clamp(boxes[:, 2],
                              min=0.0, max=width)*scale_x
    boxes[:, 1] = torch.clamp(boxes[:, 1],
                              min=0.0, max=height)*scale_y
    boxes[:, 3] = torch.clamp(boxes[:, 3],
                              min=0.0, max=height)*scale_y

    boxes, inds = eliminate_zero_area(boxes)

    return boxes, inds


def intersection(box1, box2):
    '''
    box1 Nx4
    box2 1x4
    '''
    ones = torch.ones(box1.shape[0], dtype=box1.dtype)
    xr_min = torch.min(box1[:, 2], ones*box2[2])
    xl_max = torch.max(box1[:, 0], ones*box2[0])

    yt_max = torch.max(box1[:, 1], ones*box2[1])
    yb_min = torch.min(box1[:, 3], ones*box2[3])

    x_area = xr_min - xl_max
    y_area = yb_min - yt_max

    x1 = torch.max(x_area, torch.zeros_like(x_area))
    y1 = torch.max(y_area, torch.zeros_like(y_area))

    return x1*y1


def eliminate_zero_area(boxes):
    ind = area(boxes) > 0.0

    boxes = boxes[ind, :]

    return boxes, ind

def area(boxes):
    return (boxes[:, 2] - boxes[:, 0])*(boxes[:, 3] - boxes[:, 1])