==========
User Guide
==========


.. code:: python

    import expertsegmentation as seg


Load images and training labels
--------------------------------

.. code:: python

    import tifffile
    img = tifffile.imread("path to filename")
    labels = tifffile.imread("<path to filename>")

.. note::

    Ensure that the image is a single-channel image with shape ``(H, W, [D])`` or
    ``(H, W, [D], num_features)`` for images with precomputed features.
    
    Ensure that labels are 0 for unlabeled pixels and that labels start at 1.


Create a segmentation dataset
------------------------------

To compute additional features, specify a dictionary with the desired filters. See <>
for a list of supported filters.

.. code:: python

    filters = {
        "gaussian_smoothing": [0.3, 0.7, 1, 1.6, 3.5, 5, 10],
        "laplacian_of_gaussian": [0.7, 1, 1.6, 3.5, 5, 10],
        "gaussian_gradient_magnitude": [0.7, 1, 1.6, 3.5, 5, 10],
        "diff_of_gaussians": [0.7, 1, 1.6, 3.5, 5, 10],
        "structure_tensor_eigvals": [0.7, 1, 1.6, 3.5, 5, 10],
       "hessian_of_gaussian_eigvals": [0.7, 1, 1.6, 3.5, 5, 10]
    }
    dataset = seg.SegDataset(img, labels, filters=filters, save_path="img_with_features.tif")

Or instantiate the dataset with pre-computed features:

.. code:: python

    img_with_features = tifffile.imread("<path to image with precomputed features>.tif")
    dataset = seg.SegDataset(img, labels, img_with_features=img_with_features)


Define model parameters and targets
------------------------------------------

The tuning hyperparameter lambda represents the weight of the target property
on the segmentation. Increasing lambda drives the
segmentation to weight the target more highly, but if too large
may result in a failed segmentation.

When segmenting with multiple targets, lambda can also be used to tune
the relative desired weight of each target.

If a lambda is not specified, the default is lambda=1.

No target property
^^^^^^^^^^^^^^^^^^^

Segment based on the image features alone without targeting any physical property.

.. code-block:: python

    model = seg.SegmentModel()

Volume fraction target
^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    model = seg.SegmentModel(objective='volume_fraction',
                            target={
                                1: 0.51,
                                2: 0.49,
                            },
                            lambd=1,
                            n_epochs=100)

Connectivity target
^^^^^^^^^^^^^^^^^^^^^^^
.. code:: python

    model = seg.SegmentModel(objective='connectivity',
                            direction='min',
                            target=1,
                            lambd=[0.5, 1],
                            n_epochs=100)


Volume fraction and connectivity target
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: python

    model = seg.SegmentModel(objective=['volume_fraction', 'connectivity'],
                            direction="min",
                            target={
                                'volume_fraction': {1: 0.51, 2: 0.49,},
                                'connectivity': 1,
                            },
                            lambd={'volume_fraction': [0.5, 1], 'connectivity': [1, 2]},
                            n_epochs=100)


Segment the image
------------------
.. code:: python

    model.segment(dataset)

Evaluate results
-----------------

.. code:: python

    model.plot_results(dataset)
    model.plot_steps(dataset)
    model.print_metrics()
