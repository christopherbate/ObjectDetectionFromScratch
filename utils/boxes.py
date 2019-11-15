import torch
import torchvision


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


def area(boxes):
    return (boxes[:, 2] - boxes[:, 0])*(boxes[:, 3] - boxes[:, 1])


def eliminate_zero_area(boxes):
    ind = area(boxes) > 0.0

    boxes = boxes[ind, :]

    return boxes, ind


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


def create_anchors(feature_shape: tuple,
                   image_shape: tuple,
                   anchor_size=(256, 256),
                   dtype=torch.float32,
                   device=None):
    '''
    Features shape: N x C x H x W
    Image Shape: N x c x H x W
    Anchor_size : w x h
    '''
    stride_size = torch.tensor((image_shape[0] / feature_shape[0],
                                image_shape[1]/feature_shape[1]),
                               dtype=dtype, device=device)

    width_range = torch.arange(1, feature_shape[1]+1, 1, dtype=dtype,
                               device=device)
    height_range = torch.arange(
        1, feature_shape[0]+1, 1, dtype=dtype, device=device)

    centers_y = stride_size[0]*height_range - (stride_size[0]/2)
    centers_x = stride_size[1]*width_range - (stride_size[1]/2)

    grid_y, grid_x = torch.meshgrid(centers_y, centers_x)
    centers = torch.stack([grid_x, grid_y], dim=-1)

    torch.half_anchor = torch.tensor(anchor_size, dtype=dtype, device=device)/2

    # Upper left
    ul = torch.zeros_like(centers, dtype=dtype)
    ul = centers - torch.half_anchor
    # lower right
    lr = torch.zeros_like(centers, dtype=dtype)
    lr = centers + torch.half_anchor

    anchors = torch.cat([ul, lr], dim=-1).reshape(-1, 4)

    return anchors


def anchor_box_iou(anchors, boxes):
    '''

    Anchors: M x 4 tensors
    Boxes: B x 4 tensor

    Output: M x B tensor of IoU values

    '''
    iou = torchvision.ops.box_iou(anchors, boxes)
    iou.requires_grad = False
    return iou


def create_label_matrix(iou, pos_threshold=0.5, neg_threshold=0.1):
    ''' 
    Takes the M x B tensor of IoU values 
    from anchor_box_iou and generates the 
    "object-ness" label matrix per anchor.
    '''
    # Find the best ind for each true object
    # Artificially inflate IoU so we get at least one match
    # per box.

    # This gets the index of the best anchor for each box
    vals, best_ind = iou.max(dim=0)
    col_ind = torch.arange(best_ind.shape[0])

    # Make sure that one anchor is made positive, if
    # we have a non-zero box (boxes are padded in the beginning during batching)
    iou[best_ind, col_ind] *= 10.0

    pos_mask = torch.sum(iou >= pos_threshold, dim=1, dtype=torch.bool)
    neg_mask = torch.prod(iou < neg_threshold, dim=1,
                          dtype=torch.int8).to(torch.bool)

    fgbg_mask = torch.zeros(iou.shape[0], dtype=torch.int)
    fgbg_mask[pos_mask] = 1
    fgbg_mask[neg_mask] = -1
    fgbg_mask.requires_grad = False

    _, class_assignments = torch.max(iou, dim=1)
    class_assignments = class_assignments[pos_mask]
    class_assignments.requires_grad = False

    return fgbg_mask, pos_mask, class_assignments
