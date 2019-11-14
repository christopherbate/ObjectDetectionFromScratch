# Object Detection From Scratch in PyTorch
The repo is a minimalistic implementation of a single-stage dense object detection model as pioneered by models such as SSD and RetinaNet. 

The data loader, model, and training scripts are all designed so that someone learning these sorts of systems can run the training on a CPU, even just a laptop, with 8GB of RAM. In particular, it allows you to scale imported images down to a small size (say 128x64), converted to 
greyscale, and run the model on a subset of COCO classes. 

### Presentation / Lecture Notes
I described this class of models in class on 11/13. Link to to lecture slides / presentation describing the basics of dense object detectors and the organization of this model coming soon. 

In the meantime, this [video](https://www.youtube.com/watch?v=nDPWywWRIRo&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=12&t=0s) from Stanford's CS231N course is a good 
introduction. 

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

See the `train.py` main function (at the bottom) for a list of possible arguments. For example, to train on GPU with the image resized to 256x128x1 at batch size 32, use the following:

```
python train.py --db /home/chris/datasets/coco/train.db --images /home/chris/datasets/coco/train2017 --device cuda --resize 128 256 --epochs 10 --batch_size 32
```

Note that to change the training image size, the input format should be **--resize WIDTH HEIGHT**


Currently, on GPU, it takes about 1GB of GPU space when images are resampled to `128x64x1` (WxH, greyscale), and batch size 
is 16. So, You can scale to very large batch sizes with the right GPU.

To visualize  training progress and see images, run Tensorboard from your code diretory:

```
tensorboard --logdir=./runs
```


## About the Visualizations 

When developing a **dense object detection** model like this, the **ClassHead** is a CNN which predicts 
whether or not an object is at each position in the final feature map of the **backbone**.
