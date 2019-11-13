import torch


class AnchorGenerator(torch.nn.Module):
    '''
    Adapted from torchvision anchor generator module
    Doesn't have all the features.

    We assume all images are the same size to avoid
    having to replicate anchors across images.

    simplified and renamed some things such as "cell anchors"
    to "base anchors", which seems to make more sense.

    you set sizes and aspect_ratios in the order 
    that they should appear when passed in.
    '''

    def __init__(self, sizes=((128, 256),),
                 aspect_ratios=(1.0,), **kwargs):
        super(AnchorGenerator, self).__init__(**kwargs)
        self.sizes = sizes
        self.aspect_ratios = (aspect_ratios,)*len(sizes)
        self._cache = {}
        self.base_anchors = None
        self.anchor_indices = []

    @staticmethod
    def generate_base_anchors(scales, aspect_ratios,
                              dtype=torch.float32, device="cpu"):
        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(
            aspect_ratios, dtype=dtype, device=device)
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios

        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)

        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        return base_anchors.round()

    def set_base_anchors(self, dtype, device):
        if self.base_anchors is not None:
            return self.base_anchors
        self.base_anchors = [
            self.generate_base_anchors(
                sizes,
                aspect_ratios,
                dtype,
                device
            )
            for sizes, aspect_ratios in zip(self.sizes, self.aspect_ratios)
        ]

    def grid_anchors(self, grid_sizes, strides):
        anchors = []
        for size, stride, base_anchors in zip(
            grid_sizes, strides, self.base_anchors
        ):
            grid_height, grid_width = size
            stride_height, stride_width = stride

            device = base_anchors.device
            shifts_x = torch.arange(
                0, grid_width, dtype=torch.float32, device=device
            ) * stride_width
            shifts_y = torch.arange(
                0, grid_height, dtype=torch.float32, device=device
            ) * stride_height
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            anchors.append(
                (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
            )

        return anchors

    def get_cached_anchors(self, grid_sizes, strides):
        key = tuple(grid_sizes) + tuple(strides)
        if key in self._cache:
            return self._cache[key]
        anchors = self.grid_anchors(grid_sizes, strides)
        self._cache[key] = anchors
        self.anchor_indices = [a.shape[0] for a in anchors]
        print("Num Indices: ", self.anchor_indices)
        return anchors

    def forward(self, images, feature_maps):
        '''
        images: a list or a tensor of the image batch
        feature_maps: a list of feature maps to map
            anchors to.
        '''

        # Retrieve the grid sizes.
        # We allow these to vary between batches
        image_size = images.shape[-2:]
        grid_sizes = tuple([feature_map.shape[-2:] for feature_map
                            in feature_maps])        
        strides = tuple(((float(image_size[0]) / float(g[0]),
                          float(image_size[1]) / float(g[1])))
                        for g in grid_sizes)
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        self.set_base_anchors(dtype, device)
        anchors = self.get_cached_anchors(grid_sizes, strides)
        anchors = torch.cat(anchors, dim=0)
        return anchors
