# expert-segmentation

Repo for decision tree-based semantic segmentation of microstructural images with domain knowledge-informed targets for volume fraction, connectivity, and circularity.

## Setup

Recommended to use Anaconda distribution for Python environments and package management: https://www.anaconda.com/download

Run in a terminal from the top level of the repo:
```
$ conda env create -f environment.yml
$ conda activate expert-seg
```


## Instructions

1. Hand-label image with third-party tool (Ilastik, ImageJ) and save both the raw input image and the labeled image to data folder.

        NOTE: The input image is expected to be single-channel, i.e. with shape (H, W) or (H, W, D).

        The labeled image is expected to be single-channel, i.e. with shape (H, W) or (H, W, D). It should have integer type, where 0 indicates an unlabeled pixel and every nonzero integer represents a unique class.


2. In data/user_input.json, set the domain targets and the parameters for input feature transformations.


3. Run the following command:

```
(expert-seg) $ python expert-segmentation/main.py -i <path to raw input image> -l <path to labeled input image>
```
