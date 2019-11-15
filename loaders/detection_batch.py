import torch


class ObjectDetectionBatch:
    '''
    This object wraps a single "batch" 
    consisting of images, boxes, and labels for the boxes. 

    The constructor (__init__) accepts a list "example_list", which are individual samples
    loaded from the loader (fbs_loader)

    It then then collates (batches) the data and can be passed to the model.
    '''

    def __init__(self, example_list: list):
        imgs = [ex["image"] for ex in example_list]
        boxes = [ex["boxes"] for ex in example_list]
        labels = [ex["labels"] for ex in example_list]
        labels_idx = [ex["labels_idx"] for ex in example_list]

        out = None
        if(torch.utils.data.get_worker_info() is not None):
            # This is from pytorch's default colate fn:
            # https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel_img = sum([x.numel() for x in imgs])
            storage = imgs[0].storage()._new_shared(numel_img)
            out = imgs[0].new(storage)

        # Boxes and labels use zero padding out to max length.
        # We use a util function used for RNNs, but the purpose/effect is the same.
        self.boxes = torch.nn.utils.rnn.pad_sequence(boxes, batch_first=True)
        self.labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
        self.labels_idx = torch.nn.utils.rnn.pad_sequence(
            labels_idx, batch_first=True)
        self.images = torch.stack(imgs, dim=0, out=out)

        assert(self.boxes.shape[0] ==
               self.images.shape[0] == self.labels.shape[0])

        self.debug = False

    def pin_memory(self):
        '''
        See the following link for why this is needed:
        https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading
        '''
        self.boxes = self.boxes.pin_memory()
        self.labels = self.labels.pin_memory()
        self.images = self.images.pin_memory()
        self.labels_idx = self.labels_idx.pin_memory()
        return self

    def to(self, device):
        '''
        Moves boxes, labels, and images to a given device
        '''
        self.boxes = self.boxes.to(device)
        self.labels = self.labels.to(device)
        self.images = self.images.to(device)
        self.labels_idx = self.labels_idx.to(device)


def collate_detection_samples(example_list):    
    batch = ObjectDetectionBatch(example_list)    
    return batch
