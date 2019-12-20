import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from loaders import DetectionLoader, DetectionTransform
from mpl_toolkits.axes_grid1 import ImageGrid
from utils import normalize_tensor


def plot_tensor(t, title="", normalize=True):
    grid = torchvision.utils.make_grid(
        t, padding=2, normalize=normalize, scale_each=True)
    grid = grid.permute(1, 2, 0)
    plt.imshow(grid, cmap='gray')
    plt.title(title)


def visualize_conv(conv_layer):
    fw = conv_layer.weight
    fshape = fw.shape[-1]*fw.shape[-2]
    cfshape = fshape*fw.shape[-3]

    plt.figure()
    cols = 3

    plt.subplot(4, cols, 1)
    plot_tensor(fw[:, :3, :, :])

    plt.subplot(4, cols, 2)
    gram = fw.reshape(-1, cfshape)
    gram = torch.matmul(gram, gram.T)
    plt.imshow(gram, cmap='gray')

    plot_pos = [j*cols+1 for j in range(1, 4)]
    for ch in range(3):
        plt.subplot(4, cols, plot_pos[ch])
        channel = fw[:, ch:ch+1, :, :]
        plot_tensor(channel, "Channel {}".format(ch+1))

        plt.subplot(4, cols, plot_pos[ch]+1)
        gram = channel.reshape(-1, fshape)
        gram = torch.matmul(gram, gram.T)
        plt.imshow(gram, cmap='gray')

        channel = channel.reshape(-1, fshape).numpy()
        print("Performing SVD on weights matrix {}".format(channel.shape))

        u, s, vh = np.linalg.svd(channel)

        print(u.shape, s.shape, vh.shape)

        smat = np.zeros((fw.shape[0], fshape))
        smat[:fshape, :fshape] = np.diag(s[:fshape])
        svt = np.dot(smat, vh)
        reconstructed = np.dot(u, svt)

        print("Singular values:")
        print(s)

        plt.subplot(4, cols, plot_pos[ch]+2)
        channel = torch.from_numpy(
            vh).reshape(-1, 1, fw.shape[-2], fw.shape[-1])
        plot_tensor(channel, "Reconstructed")
    plt.show()


def get_weights_simplified(weight, top_n=10, channel=0):
    channel = weight[:, channel:channel+1, :, :]

    rs = channel.reshape(weight.shape[0], -1).numpy()
    u, s, vh = np.linalg.svd(rs)

    print(vh.shape)
    return torch.from_numpy(vh[:10, :]).reshape((top_n, 1, weight.shape[-2], weight.shape[-1]))


def upsample_filter(weight, size=256):
    weight_up = torch.nn.functional.interpolate(
        weight[None, None, :, :], size=(size, size), mode='nearest')
    return weight_up[0, 0]


def plot_spectrum_filter(weight, shape=[256, 256]):
    num_filters = weight.shape[0]
    cols = num_filters // 10
    fig = plt.figure(figsize=(10, 10))
    filter_spectrums = np.zeros([num_filters]+shape)
    filter_idx = 0
    for col_idx in range(cols):
        grid = ImageGrid(fig, (1, cols, col_idx+1), nrows_ncols=(
            10, 2), axes_pad=0.1, share_all=True, add_all=True, cbar_mode='single')
        for idx in range(10):
            if filter_idx > num_filters-1:
                break

            weight_up = torch.nn.functional.interpolate(
                weight[None, filter_idx:filter_idx+1, :, :], size=256, mode='nearest')
            im = grid[2*idx].imshow(weight_up[0, 0], cmap='gray')

            ff = np.fft.fft2(weight[filter_idx], s=shape)
            ff = np.fft.fftshift(ff)
            ff = np.abs(ff)
            ff = ff/ff.max()
            ff = np.where(ff < 0.60, np.ones_like(ff)*1e-10, ff)
            im = grid[2*idx+1].imshow(ff, cmap='gray')
            filter_spectrums[filter_idx] = ff
            filter_idx += 1

        grid.cbar_axes[0].colorbar(im)
    plt.show()

    overlayed_spectrum = filter_spectrums.mean(axis=0)
    print(overlayed_spectrum.shape)

    im = plt.imshow(overlayed_spectrum, cmap='gray')
    plt.colorbar(im)
    plt.show()


def get_spectrum(features, ax=None, shape=None, truncate=None):
    ff = np.fft.fft2(features, s=shape, norm="ortho")
    ff = np.fft.fftshift(ff)
    return ff


def viz_spectrum(ff, truncate=None):
    viz = np.abs(ff)
    viz = 20*np.log10(viz/viz.max())
    return viz


def visualize_filter_effect(image, weight, weight2=None):
    '''
    Plots
    1 - filter 2 - filter (fourier)
    3 - img conv 4  img conv (fourier)
    5 - img conv + relu (fourier)
    '''
    fig = plt.figure()
    grid = ImageGrid(fig, 111, nrows_ncols=(6, 2), axes_pad=0.1,
                     share_all=True, add_all=True, cbar_mode='single')

    up = upsample_filter(weight, size=256)
    print("Upsampled shape ", up.shape)
    grid[0].imshow(up, cmap='gray')

    filter_fourier = get_spectrum(weight, shape=[256, 256])
    grid[1].imshow(viz_spectrum(filter_fourier), cmap='gray')

    features = torch.nn.functional.conv2d(
        image.unsqueeze(0).unsqueeze(0), weight.unsqueeze(0).unsqueeze(0), padding=3, stride=1)
    features = features[0, 0]

    grid[2].imshow(image, cmap='gray')
    img_fourier = get_spectrum(image)
    grid[3].imshow(viz_spectrum(img_fourier), cmap='gray')

    grid[4].imshow(features, cmap='gray')
    features_fourier = get_spectrum(features)
    grid[5].imshow(viz_spectrum(features_fourier), cmap='gray')

    features_relu = torch.relu(features)
    grid[6].imshow(features_relu, cmap='gray')

    fr_fourier = get_spectrum(features_relu)
    grid[7].imshow(viz_spectrum(fr_fourier), cmap='gray')

    if weight2 is not None:
        filter_fourier = get_spectrum(weight2, shape=[256, 256])
        grid[9].imshow(viz_spectrum(filter_fourier), cmap='gray')

        features2 = torch.nn.functional.conv2d(features_relu.unsqueeze(0).unsqueeze(0),
                                               weight2.unsqueeze(
                                                   0).unsqueeze(0),
                                               padding=weight2.shape[-1]//2, stride=1)
        features2 = features2[0, 0]
        grid[10].imshow(features2, cmap='gray')

        features2_fourier = get_spectrum(features2)
        grid[11].imshow(viz_spectrum(features2_fourier), cmap='gray')

    plt.show()


def experiment():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True, progress=True, pretrained_backbone=True)

    body = model.backbone.body
    print(body)

    # plot_spectrum_filter(body.conv1.weight[:, 0, :, :])
    # plot_spectrum_filter(body.layer1[0].conv2.weight[:, 0, :, :])
    # plot_spectrum_filter(body.layer2[0].conv2.weight[:, 0, :, :].detach())
    # plot_spectrum_filter(body.layer3[0].conv2.weight[:, 0, :, :].detach())

    transforms = DetectionTransform(output_size=(256, 256),
                                    greyscale=True, normalize=True)
    loader = DetectionLoader(
        "/home/chris/coco_tools/test.db", "/home/chris/datasets/coco/train2017/",
        area_filter=None, categories_filter=["person"],
        greyscale=True, transforms=transforms)

    for idx, sample in enumerate(loader):
        img = sample['image'][0, :, :]

        for idx in range(body.conv1.weight.shape[0]):
            w1 = body.conv1.weight[idx, 0, :, :]
            w2 = body.layer1[0].conv2.weight[0, 0, :, :].detach()
            visualize_filter_effect(img, w1, w2)
        break


if __name__ == '__main__':
    with torch.no_grad():
        experiment()
