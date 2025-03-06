"""
Functionality for data handling including loading and adding image transformation features.
"""

import numpy as np
from scipy import ndimage
from skimage import filters, feature
from structure_tensor import (
    eig_special_2d,
    structure_tensor_2d,
    eig_special_3d,
    structure_tensor_3d,
)
import tifffile
from typing import Union

from expertsegmentation.named_constants import DEFAULT_FILTERS


def load_example_data():
    """
    For convenience, load example data.
    Returns:
        A (75,75,75) NMC microCT image.
        Sparse labels on the image.
        The image with additional channels for transformation features,
        computed using the defaults.
    """
    img = tifffile.imread("example_data/nmc1cal.tif")
    labels = tifffile.imread("example_data/nmc1cal_labels.tiff")
    img_with_features = tifffile.imread("example_data/nmc1cal_with_features.tiff")

    return SegDataset(img=img, labeled_img=labels, img_with_features=img_with_features)


class SegDataset:
    """Class responsible for organizing and preprocessing data.

        Args:
            img (np.ndarray):                Input image .tif.
                                             Should be 2 or 3D, single-channel
            labeled_img (np.ndarray):        Filepath to labeled input image.
                                Should be dtype integer with 0 = unlabeled
            img_with_features:  (Optional) Input image with
                                features already computed at axis=-1 (shape (H, W, [D], num_features)). If provided,
                                features are not re-computed.
            filters:            Dictionary defining which filters to apply
            save_path:            If provided, filepath to save image with
                                computed features.

        Returns:
            None
        """
    def __init__(
        self,
        img: np.ndarray,
        labeled_img: np.ndarray,
        img_with_features: Union[np.ndarray, None] = None,
        filters: dict = None,
        save_path: str = None,
    ):

        self.img = img
        self.labeled_img = labeled_img
        self.filters = filters
        self.save_fn = save_path

        if labeled_img.shape != img.shape:
            raise ValueError(f"Expected img and labels to have the same dimensions.")

        if img_with_features is not None:
            self.img_with_features = img_with_features
            if img_with_features.shape[:-1] != img.shape:
                raise ValueError(f"Expected img and labels to have the same dimensions up to last channel of img_with_features.")
        
        else:
            if filters is None:
                self.filters = DEFAULT_FILTERS
            print('Computing additional features...')
            self.add_filter_features()

        self.n_classes = len(np.unique(self.labeled_img)) - 1  # Excluding 0

    def add_filter_features(self):
        """Compute image transformations and stack them onto last dimension.
        
        Takes single-channel input image and constructs array with filters
        of the image stacked onto the channel dimension, i.e. with shape
        (H, W, n_filters+1) or (H, W, D, n_filters+1) where the first
        channel is the original unfiltered image.

        Called automatically when instantiating the dataset, and `img_with_features`
        is available as an attribute once it's called.

        Supported filters:
        - 'gaussian_smoothing'
        - 'laplacian_of_gaussian'
        - 'gaussian_gradient_magnitude'
        - 'diff_of_gaussians'
        - 'structure_tensor_eigvals'
        - 'hessian_of_gaussian_eigvals'

        """

        # Add a channel to the end to stack onto
        result = np.expand_dims(self.img, -1)

        # Set up a function to make sure that each filter matches
        # the original dtype
        if result.dtype == np.uint8:
            rescale_f = self.rescale_to_uint8
        elif result.dtype == np.uint16:
            rescale_f = self.rescale_to_uint16

        for filter in self.filters:
            sigmas = self.filters[filter]
            if filter == "gaussian_smoothing":
                for sigma in sigmas:
                    img_filter = filters.gaussian(self.img, sigma=sigma)
                    img_filter = rescale_f(img_filter)
                    result = np.concatenate(
                        [result, np.expand_dims(img_filter, -1)], axis=-1
                    )
            elif filter == "laplacian_of_gaussian":
                for sigma in sigmas:
                    # output has to allow signed (derivatives can be negative)
                    img_filter = ndimage.gaussian_laplace(
                        self.img, sigma=sigma, output=np.int16
                    )
                    img_filter = rescale_f(img_filter)
                    result = np.concatenate(
                        [result, np.expand_dims(img_filter, -1)], axis=-1
                    )
            elif filter == "gaussian_gradient_magnitude":
                for sigma in sigmas:
                    # output has to allow signed (derivatives can be negative)
                    img_filter = ndimage.gaussian_gradient_magnitude(
                        self.img, sigma=sigma, output=np.int16
                    )
                    img_filter = rescale_f(img_filter)
                    result = np.concatenate(
                        [result, np.expand_dims(img_filter, -1)], axis=-1
                    )
            elif filter == "diff_of_gaussians":
                for sigma in sigmas:
                    img_filter = filters.difference_of_gaussians(
                        self.img, low_sigma=sigma
                    )
                    img_filter = rescale_f(img_filter)
                    result = np.concatenate(
                        [result, np.expand_dims(img_filter, -1)], axis=-1
                    )
            elif filter == "structure_tensor_eigvals":
                for sigma in sigmas:
                    # Structure tensor wants float type
                    if self.img.ndim == 2:
                        S = structure_tensor_2d(
                            self.img.astype(np.float32), sigma=sigma, rho=1
                        )
                        img_filter = eig_special_2d(S)[0][1]
                        img_filter = self.rescale_to_uint8(img_filter.astype(np.uint32))
                    elif self.img.ndim == 3:
                        S = structure_tensor_3d(
                            self.img.astype(np.float32), sigma=sigma, rho=1
                        )
                        img_filter = eig_special_3d(S)[0][1]
                        img_filter = self.rescale_to_uint8(img_filter.astype(np.uint32))
                    result = np.concatenate(
                        [result, np.expand_dims(img_filter, -1)], axis=-1
                    )
            elif filter == "hessian_of_gaussian_eigvals":
                for sigma in sigmas:
                    hessian = feature.hessian_matrix(
                        self.img, sigma=sigma, use_gaussian_derivatives=True
                    )
                    img_filter = feature.hessian_matrix_eigvals(hessian)[0]
                    img_filter = self.rescale_to_uint8(img_filter)
                    result = np.concatenate(
                        [result, np.expand_dims(img_filter, -1)], axis=-1
                    )
            else:
                str_list = set(
                    [
                        "gaussian_smoothing",
                        "diff_of_gaussians",
                        "hessian_of_gaussian_eigvals",
                        "laplacian_of_gaussian",
                        "gaussian_gradient_magnitude",
                        "structure_tensor_eigvals",
                    ]
                )
                raise ValueError(
                    f"Filter should be one of {str_list}. Received {filter}"
                )

        self.img_with_features = result
        if self.save_fn is not None:
            tifffile.imwrite(
                f"{self.save_fn.split('.')[0]}_with_features.tiff", result
            )

    @staticmethod
    def rescale_to_uint8(img: np.ndarray):
        """
        Rescale the values of an array to the range (0, 255)

        Args:
            img (np.ndarray) Image to rescale

        Returns:
            img_uint8 (np.ndarray)  Image after rescaling
        """
        eps = 1e-6
        img_norm = (img - img.min()) / (img.max() - img.min() + eps)
        img_uint8 = (img_norm * 255).astype(np.uint8)
        return img_uint8

    @staticmethod
    def rescale_to_uint16(img: np.ndarray):
        """
        Rescale the values of an array to the range (0, 65535)

        Args:
            img (np.ndarray) Image to rescale

        Returns:
            img_uint16 (np.ndarray) Image after rescaling
        """
        eps = 1e-6
        img_norm = (img - img.min()) / (img.max() - img.min() + eps)
        img_uint16 = (img_norm * 65535).astype(np.uint16)
        return img_uint16
