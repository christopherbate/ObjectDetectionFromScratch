import torch


def normalize_tensor(tensor, range=None):
    tensor = tensor.clone()  # avoid modifying tensor in-place
    if range is not None:
        assert isinstance(range, tuple), \
            "range has to be a tuple (min, max) if specified. min and max are numbers"

    def norm_ip(img, min, max):
        img.clamp_(min=min, max=max)
        img.add_(-min).div_(max - min + 1e-5)

    def norm_range(t, range):
        if range is not None:
            norm_ip(t, range[0], range[1])
        else:
            norm_ip(t, float(t.min()), float(t.max()))
    norm_range(tensor, range)
    return tensor
