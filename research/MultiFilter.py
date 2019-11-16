import torch
import numpy as np
import math
from scipy import ndimage
import scipy.stats
from research.gauss import gaussian2d

def gauss_initializer(filter_tensor):
    width = filter_tensor.shape[-1]
    height = filter_tensor.shape[-2]
    intermediate_channels = filter_tensor.shape[0]
    sigma_x = torch.linspace(
        width/4, width/2, intermediate_channels)
    sigma_y = torch.linspace(
        height/4, height/2, intermediate_channels)
    for gauss_idx in range(intermediate_channels//2):
        gauss = gaussian2d(1, kernel_size=(height, width),
                           sigma=(
            sigma_x[gauss_idx], sigma_y[gauss_idx]),
            offset=(0, 0))
        gauss = gauss.reshape(1, height, width)
        filter_tensor[gauss_idx] = gauss

    for chn_idx, gauss_idx in enumerate(range(intermediate_channels//2,
                                              intermediate_channels)):
        gauss1 = gaussian2d(1, kernel_size=(height, width),
                            sigma=(
            sigma_x[chn_idx], sigma_y[chn_idx]),
            offset=(0, 0))
        gauss2 = gaussian2d(1, kernel_size=(height, width),
                            sigma=(
            sigma_x[chn_idx]/2, sigma_y[chn_idx]/2),
            offset=(0, 0))
        filter_tensor[gauss_idx] = gauss2-gauss1
    return filter_tensor


def gkern(size=(21, 21), offset=(0, 0), scale=(1, 1), nsig=3):
    """Returns a 2D Gaussian kernel."""
    x = np.linspace(-nsig, nsig, size[0]+1)
    y = np.linspace(-nsig, nsig, size[1]+1)
    kernX = np.diff(scipy.stats.norm.cdf(x, loc=offset[0], scale=scale[0]))
    kernY = np.diff(scipy.stats.norm.cdf(y, loc=offset[1],  scale=scale[1]))
    kern2d = np.outer(kernX, kernY)
    return kern2d


def zero_order_initializer(filter_tensor, sigma=(2, 1)):
    width = filter_tensor.shape[-1]
    height = filter_tensor.shape[-2]
    num_output = filter_tensor.shape[0]
    num_input = filter_tensor.shape[1]
    base_filter = torch.zeros_like(filter_tensor[0, 0])
    base_filter += torch.from_numpy(gkern((height, width),
                                          offset=(0, 0), scale=sigma,
                                          nsig=3))
    angles = torch.linspace(0, 360, num_output)
    for chn_idx in range(num_output):
        sigma = 1.0*2**(chn_idx/num_output)
        for in_idx in range(num_input):
            filter_tensor[chn_idx, in_idx] = torch.from_numpy(ndimage.rotate(base_filter.numpy(),
                                                                             angles[chn_idx],
                                                                             reshape=False, mode='nearest'))
    return filter_tensor


def first_order_initializer(filter_tensor):
    with torch.no_grad():
        width = filter_tensor.shape[-1]
        height = filter_tensor.shape[-2]
        num_output = filter_tensor.shape[0]
        num_input = filter_tensor.shape[1]
        sigma = (height/3, height/3)
        offset_1 = 1.0
        offset_2 = height/2
        base_filter = torch.zeros_like(filter_tensor[0, 0])
        base_filter += torch.from_numpy(gkern((height, width),
                                            offset=(offset_1, 0), scale=(1, 1),
                                            nsig=3))

        base_filter -= torch.from_numpy(gkern((height, width),
                                            offset=(-offset_1, 0), scale=(1, 1),
                                            nsig=3))

        angles = torch.linspace(0, 360, num_output)
        for chn_idx in range(num_output):
            for in_idx in range(num_input):
                filter_tensor[chn_idx, in_idx] = torch.from_numpy(ndimage.rotate(base_filter.numpy(),
                                                                                angles[chn_idx],
                                                                                reshape=False, mode='nearest'))
        return filter_tensor


def blob_initializer(filter_tensor):
    with torch.no_grad():
        width = filter_tensor.shape[-1]
        height = filter_tensor.shape[-2]
        num_output = filter_tensor.shape[0]
        num_input = filter_tensor.shape[1]
        sigma = (height/3, height/3)
        offset_1 = 1.0
        offset_2 = height/2

        angles = torch.arange(0, 360, num_output)
        for chn_idx in range(num_output):
            sigma = 1.0
            sigma_prev = 1.7
            for in_idx in range(num_input):
                base_filter = torch.zeros_like(filter_tensor[0, 0])
                base_filter += torch.from_numpy(gkern((height, width),
                                                    offset=(0, 0), scale=(sigma, sigma),
                                                    nsig=3))
                base_filter -= 2.0*torch.from_numpy(gkern((height, width),
                                                    offset=(0, 0), scale=(sigma_prev, sigma_prev),
                                                    nsig=3))
                filter_tensor[chn_idx, in_idx] = base_filter

        return filter_tensor