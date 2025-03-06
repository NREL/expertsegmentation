# ExpertSegmentation

ExpertSegmentation is a decision tree-based semantic segmentation tool for microstructural images with domain knowledge-informed physical targets for the resulting segmentation.

Supported targets include volume fractions and minimum or maximum phase connectivity.

* MIT License
* Docs: https://expertsegmentation.readthedocs.io/en/latest/index.html


## Setup

Recommended to use [Anaconda distribution](https://www.anaconda.com/download) for Python environments and package management.

After cloning the [expertsegmentation repo](https://github.com/NREL/expertsegmentation) navigate to the top level:

```
$ conda env create -f environment.yml
$ conda activate expert-seg
$ pip install .
```

## Quickstart

Hand-label image with third-party tool ([Ilastik](https://www.ilastik.org/documentation/basics/installation), the [SAMBA web-based tool](https://www.sambasegment.com/)) and save both the raw input image and the labeled image.

Ensure that the labeled image file has the same dimensions as the raw input image, and that it has integer datatype with labels = 0 for unlabeled pixels with phase labels enumerated starting at 1. See docs for more details.

```
import expertsegmentation as seg
```


### Load data

```
import tifffile
img = tifffile.imread("path to filename")
labels = tifffile.imread("<path to filename>")
```


### Create a segmentation dataset

```
# Option 1
dataset = seg.SegDataset(img, labels)

# Option 2: Define custom filters to use as additional features. The example below is the default set of filters.
dataset = seg.SegDataset(img, labels, filters = {
    "gaussian_smoothing": [0.3, 0.7, 1, 1.6, 3.5, 5, 10],
    "laplacian_of_gaussian": [0.7, 1, 1.6, 3.5, 5, 10],
    "gaussian_gradient_magnitude": [0.7, 1, 1.6, 3.5, 5, 10],
    "diff_of_gaussians": [0.7, 1, 1.6, 3.5, 5, 10],
    "structure_tensor_eigvals": [0.7, 1, 1.6, 3.5, 5, 10],
   "hessian_of_gaussian_eigvals": [0.7, 1, 1.6, 3.5, 5, 10]
})

# Option 3: Provide an image with pre-computed features stored in the last image channel (H, W, D, num_features).
dataset = seg.SegDataset(img, labels, img_with_features=img_with_features)

# Option 4: Load example data
dataset = seg.load_example_data()

```

### Set up the segmentation model and targets

The tuning hyperparameter lambda represents the weight of the target property
on the segmentation. Increasing lambda drives the
segmentation to weight the target more highly, but if too large
may result in a failed segmentation.

When segmenting with multiple targets, lambda can also be used to tune
the relative desired weight of each target.

If a lambda is not specified, the default is lambda=1.

```
# No target property
model = seg.SegmentModel()

# Volume fraction target (51% on the class labeled 1, 49% on class 2)
model = seg.SegmentModel(objective='volume_fraction',
                        target={
                        1: 0.51,
                        2: 0.49,
                        },
                        lambd=1,
                        n_epochs=100)
```

### Segment the image
```
model.segment(dataset)
```

### Evaluate and visualize results
```
model.plot_results(dataset)
model.plot_steps(dataset)
model.print_metrics()
```

## Limitations
As of now, ExpertSeg requires that the target be fully defined; i.e. a value needs to be specified for every phase in the image of interest such that the volume fractions sum to 1. The ability to specify one or a subset of volume fractions for multiphase materials is currently not supported by may be implemented in the future.
