# Object Detection From Scratch in PyTorch
The repo is a minimalistic implementation of a single-stage dense object detection model as pioneered by models 
such as SSD and RetinaNet. 

### Presentation / Lecture Notes
Link to to lecture slides / presentation describing the basics of dense object detectors and the organization of this model coming soon.

# Getting Started

To use this code, you will need a local copy of:
1. COCO Train 2017 Images
2. Annotations 

**COCO**: Download it from the COCO website, and unzip it in a folder, preferably
`/home/[user]/datasets/coco/train2017`.

**Annotations**: We reorganized the annotations into a Flatbuffer file (schema is in the loaders directory), download [here](https://drive.google.com/file/d/1XF2iDwn8S_d8-nrDtiLvKpH2JEcD9voL/view?usp=sharing).
In the future we will be transferring to SQLite3 database file for storing annotations. These are much easier to filter for 
special training scenarios. 

Place the annotations `train.db` in the same folder as this repo.


## Installing PyTorch

You should install the latest version of PyTorch according to the simple instructions on their [website](www.pytorch.org).

To use Tensorboard, you also need to install Tensorflow (either 2.0 or the latest 1.x is fine). Do so according to the simple instructions 
on their [website](www.pytorch.org)

Of course, if you have an NVIDIA GPU, you should install the GPU versions, otherwise install the CPU versions.

## Training 

For CPU only:

```
train.py --db ./train.db --images /home/[user]/datasets/coco/train2017
```

For GPU:

```
train.py --db ./train.db --images /home/[user]/datasets/coco/train2017 --device cuda
```


Currently, on GPU, it takes about 1GB of GPU space when images are resampled to `128x64x1` (WxH, greyscale).

To visualize  training progress and see images, run Tensorboard from your code diretory:

```
tensorboard --logdir=./runs
```
