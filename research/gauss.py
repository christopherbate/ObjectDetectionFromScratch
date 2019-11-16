import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2, stride=1, padding=0):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        self.stride = stride
        self.padding = padding
        if(padding == 'same'):
            self.padding = (math.floor(kernel_size[0]/2),
                            math.floor(kernel_size[1]/2))
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
        kernel.requires_grad = False

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(
                    dim)
            )

    def forward(self, x):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        # pi = F.pad(input, (2, 2, 2, 2), mode='reflect')
        return self.conv(x, weight=self.weight,
                         groups=x.shape[1],
                         stride=self.stride,
                         padding=self.padding)


class Laplace(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Laplace, self).__init__(**kwargs)
        self.scales = [2, 3]
        self.gauss1 = GaussianSmoothing(channels=1,
                                        kernel_size=(5, 5),
                                        sigma=(1, 1),
                                        dim=2,
                                        stride=1, padding='same')
        self.gauss2 = GaussianSmoothing(channels=1,
                                        kernel_size=(5, 5),
                                        sigma=(1, 1),
                                        dim=2,
                                        stride=1, padding='same')

    def forward(self, image):
        lp1 = self.gauss2(image)
        lp2 = self.gauss1(lp1)
        diff12 = lp1-lp2
        diff1 = torch.cat([torch.relu(diff12),
                           torch.relu(-diff12),
                           torch.abs(diff12),
                           diff12.max()-torch.abs(diff12)], dim=1)

        lp3 = self.gauss2(lp2)
        lp4 = self.gauss1(lp3)
        diff34 = lp3-lp4
        diff2 = torch.cat([torch.relu(diff34),
                           torch.relu(-diff34),
                           torch.abs(diff34),
                           1-torch.abs(diff34)], dim=1)

        lp5 = self.gauss2(lp4)
        lp6 = self.gauss1(lp5)
        diff56 = lp5-lp6
        diff3 = torch.cat([torch.relu(diff56),
                           torch.relu(-diff56),
                           torch.abs(diff56),
                           1-torch.abs(diff56)], dim=1)

        lp7 = self.gauss2(lp6)
        lp8 = self.gauss1(lp7)
        diff78 = lp7-lp8
        diff78a = torch.abs(diff78)
        diff4 = torch.cat([torch.relu(diff78),
                           torch.relu(-diff78),
                           diff78a,
                           diff78a.max() - diff78a], dim=1)

        lp9 = self.gauss2(lp8)
        lp10 = self.gauss1(lp9)
        diff90 = lp9-lp10
        diff5 = torch.cat([torch.relu(diff90),
                           torch.relu(-diff90),
                           torch.abs(diff90),
                           diff90.max()-torch.abs(diff90)], dim=1)

        fm = [diff1, diff2, diff3, diff4, diff5]

        img = [lp1, lp3, lp5, lp7, lp9]

        return img, fm


def gaussian2d(channels, kernel_size=(10, 10), sigma=(3, 3), offset=(0, 0), device=torch.device("cpu")):
    # The gaussian kernel is the product of the
    # gaussian function of each dimension.
    kernel = 1
    meshgrids = torch.meshgrid(
        [
            torch.arange(size, dtype=torch.float32, device=device)
            for size in kernel_size
        ]
    )
    for size, std, mgrid, offt in zip(kernel_size, sigma, meshgrids, offset):
        center = (size - 1) / 2
        kernel *= (1 / (std * math.sqrt(2 * math.pi))) * \
            torch.exp(-((mgrid - center-offt) / std) ** 2 / 2)

    # Make sure sum of values in gaussian kernel equals 1.
    kernel = kernel / torch.sum(kernel)

    # Reshape to depthwise convolutional weight
    kernel = kernel.view(1, 1, *kernel.size())
    kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
    # kernel.requires_grad = False
    return kernel


class LearnableGaussian(torch.nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 sigma_init=[1.0, 1.0],
                 kernel_size=[5, 5],
                 stride=(1, 1),
                 padding=(0, 0),
                 groups=1, **kwargs):
        super(LearnableGaussian, self).__init__(**kwargs)
        sigma = torch.tensor(sigma_init)
        self.sigma = torch.nn.Parameter(sigma, requires_grad=True)
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.stride = stride
        self.groups = groups
        kernel = self.create_kernel()

        self.register_buffer('weight', kernel)
        print("Gaussian weight buffer: ", kernel.shape)

    def create_kernel(self, device=torch.device("cpu")):
        kernel = torch.zeros(
            (self.out_channels, self.in_channels,
             self.kernel_size[0], self.kernel_size[1]), device=device)

        for idx in range(self.out_channels):
            k = gaussian2d(self.in_channels, kernel_size=self.kernel_size,
                           sigma=self.sigma, offset=(0, 0), device=device)
            kernel[idx] = k[:, 0, :, :]
        return kernel

    def forward(self, x):
        self.weight = self.create_kernel(device=self.weight.device)

        out = torch.nn.functional.conv2d(
            x, self.weight, bias=None, stride=self.stride, padding=self.padding,
            groups=self.groups
        )

        return out


class LearnableGaussianDifference(torch.nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 sigma_init_1=[1.0, 1.0],
                 sigma_init_2=[1.0, 1.0],
                 kernel_size=[5, 5],
                 stride=(1, 1),
                 padding=(0, 0),
                 groups=1, **kwargs):
        super(LearnableGaussianDifference, self).__init__(**kwargs)
        sigma1 = torch.tensor(sigma_init_1)
        sigma2 = torch.tensor(sigma_init_2)
        sigma1 += torch.rand_like(sigma1)
        sigma2 += torch.rand_like(sigma2)
        self.sigma1 = torch.nn.Parameter(sigma1, requires_grad=True)
        self.sigma2 = torch.nn.Parameter(sigma2, requires_grad=True)
        self.offfset1 = [
            torch.randn(2) for c in range(out_channels)
        ]        
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.stride = stride
        self.groups = groups
        kernel = self.create_kernel()

        self.register_buffer('weight', kernel)
        print("Gaussian weight buffer: ", kernel.shape)

    def create_kernel(self, device=torch.device("cpu")):
        kernel = torch.zeros(
            (self.out_channels, self.in_channels,
             self.kernel_size[0], self.kernel_size[1]), device=device)

        for idx in range(self.out_channels):
            k1 = gaussian2d(self.in_channels, kernel_size=self.kernel_size,
                            sigma=self.sigma1, offset=self.offfset1[idx], device=device)
            k2 = gaussian2d(self.in_channels, kernel_size=self.kernel_size,
                            sigma=self.sigma2, offset=(0, 0), device=device)
            kernel[idx] = (k1-k2)[:, 0, :, :]
        return kernel

    def forward(self, x):
        self.weight = self.create_kernel(device=self.weight.device)

        out = torch.nn.functional.conv2d(
            x, self.weight, bias=None, stride=self.stride, padding=self.padding,
            groups=self.groups
        )

        return out
