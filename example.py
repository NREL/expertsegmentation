import expertsegmentation as seg

import tifffile
img = tifffile.imread("example_data/nmc1cal.tif")
labels = tifffile.imread("example_data/nmc1cal_labels.tiff")
img_with_features = tifffile.imread("example_data/nmc1cal_with_features.tiff")


# Option 1: Use dataset as-is. By default, additional filters are added
# as features and saved.
dataset = seg.SegDataset(img, labels)

# Option 2: Define custom filters to use as additional features.
dataset = seg.SegDataset(img, labels, filters = {
    "gaussian_smoothing": [0.3, 0.7, 1, 1.6, 3.5, 5, 10],
    "laplacian_of_gaussian": [0.7, 1, 1.6, 3.5, 5, 10],
    "gaussian_gradient_magnitude": [0.7, 1, 1.6, 3.5, 5, 10],
    "diff_of_gaussians": [0.7, 1, 1.6, 3.5, 5, 10],
    "structure_tensor_eigvals": [0.7, 1, 1.6, 3.5, 5, 10],
   "hessian_of_gaussian_eigvals": [0.7, 1, 1.6, 3.5, 5, 10]
})

# Option 3: Provide image with pre-computed features.
dataset = seg.SegDataset(img, labels, img_with_features=img_with_features)

# Option 4: Load example data
dataset = seg.load_example_data()

# Volume fraction target (51% on the class labeled 1, 49% on class 2)
model = seg.SegmentModel(objective='volume_fraction',
                        target={
                        1: 0.51,
                        2: 0.49,
                        },
                        lambd=1,
                        n_epochs=100)
model.segment(dataset)

model.plot_results(dataset)
model.plot_steps(dataset)
model.print_metrics()
