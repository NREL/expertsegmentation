"""
All functionality for data handling including loading and adding image transformation features.
"""

import json
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


def load_json(fn):
    with open(fn, "r") as f:
        result = json.load(f)
    return result


class UserInputs:
    def __init__(self, user_input_fn):
        self.user_input_dict = load_json(user_input_fn)
        self.filters = self.user_input_dict["filters"]
        self.volume_fraction_targets = np.array(
            list(self.user_input_dict["volume_fraction_targets"].values())
        )
        self.connectivity_target = self.user_input_dict["connectivity_target"]
        self.circularity_target = self.user_input_dict["circularity_target"]
        self.lambdas = self.user_input_dict["lambdas"]


class SegDataset:
    def __init__(
        self,
        raw_img_fn: str,
        labeled_img_fn: str,
        filter_dict: dict,
        raw_img_with_features_fn: [str, None],
    ):
        """
        Class responsible for organizing and preprocessing data.

        Args:
            raw_img_fn:                 Filepath to input image .tif.
                                        Should be 2 or 3D, single-channel
            labeled_img_fn:             Filepath to labeled input image.
                                        Should be integer type with 0 = unlabeled
            filter_dict:                Dictionary defining which filters to apply
            raw_img_with_features_fn:   (Optional) Filepath to input image with
                                        features already computed. If provided,
                                        features are not re-computed.
        """

        self.raw_img = tifffile.imread(raw_img_fn)
        self.labeled_img = tifffile.imread(labeled_img_fn)
        self.filter_dict = filter_dict

        if raw_img_with_features_fn is not None:
            self.raw_img_with_features = tifffile.imread(raw_img_with_features_fn)
        else:
            self._add_filter_features()

        self.n_classes = len(np.unique(self.labeled_img)) - 1  # Excluding 0

    def _add_filter_features(self):
        """
        Args:
            img:                Single-channel image with shape (H, W)
            filter_dict:        Dict of filters where keys are one or more of
                                    {'gaussian_smoothing',
                                    'diff_of_gaussians',
                                    'hessian_of_gaussian_eigvals',
                                    'laplacian_of_gaussian',
                                    'gaussian_gradient_magnitude',
                                    'structure_tensor_eigvals'} and values are
                                    lists of sigmas to apply per filter

        Returns:
            Array with filters of the image stacked onto the channel dimension,
            i.e. with shape (H, W, n_filters+1) where the first channel is the
            original unfiltered image.
        """

        # Add a channel to the end to stack onto
        result = np.expand_dims(self.raw_img, -1)

        # Set up a function to make sure that each filter matches
        # the original dtype
        if result.dtype == np.uint8:
            rescale_f = self.rescale_to_uint8
        elif result.dtype == np.uint16:
            rescale_f = self.rescale_to_uint16

        for filter in self.filter_dict:
            sigmas = self.filter_dict[filter]
            if filter == "gaussian_smoothing":
                for sigma in sigmas:
                    img_filter = filters.gaussian(self.raw_img, sigma=sigma)
                    img_filter = rescale_f(img_filter)
                    result = np.concatenate(
                        [result, np.expand_dims(img_filter, -1)], axis=-1
                    )
            elif filter == "laplacian_of_gaussian":
                for sigma in sigmas:
                    # output has to allow signed (derivatives can be negative)
                    img_filter = ndimage.gaussian_laplace(
                        self.raw_img, sigma=sigma, output=np.int16
                    )
                    img_filter = rescale_f(img_filter)
                    result = np.concatenate(
                        [result, np.expand_dims(img_filter, -1)], axis=-1
                    )
            elif filter == "gaussian_gradient_magnitude":
                for sigma in sigmas:
                    # output has to allow signed (derivatives can be negative)
                    img_filter = ndimage.gaussian_gradient_magnitude(
                        self.raw_img, sigma=sigma, output=np.int16
                    )
                    img_filter = rescale_f(img_filter)
                    result = np.concatenate(
                        [result, np.expand_dims(img_filter, -1)], axis=-1
                    )
            elif filter == "diff_of_gaussians":
                for sigma in sigmas:
                    img_filter = filters.difference_of_gaussians(
                        self.raw_img, low_sigma=sigma
                    )
                    img_filter = rescale_f(img_filter)
                    result = np.concatenate(
                        [result, np.expand_dims(img_filter, -1)], axis=-1
                    )
            elif filter == "structure_tensor_eigvals":
                for sigma in sigmas:
                    # Structure tensor wants float type
                    if self.raw_img.ndim == 2:
                        S = structure_tensor_2d(
                            self.raw_img.astype(np.float32), sigma=sigma, rho=1
                        )
                        img_filter = eig_special_2d(S)[0][1]
                        img_filter = self.rescale_to_uint8(img_filter.astype(np.uint32))
                    elif self.raw_img.ndim == 3:
                        S = structure_tensor_3d(
                            self.raw_img.astype(np.float32), sigma=sigma, rho=1
                        )
                        img_filter = eig_special_3d(S)[0][1]
                        img_filter = self.rescale_to_uint8(img_filter.astype(np.uint32))
                    result = np.concatenate(
                        [result, np.expand_dims(img_filter, -1)], axis=-1
                    )
            elif filter == "hessian_of_gaussian_eigvals":
                for sigma in sigmas:
                    hessian = feature.hessian_matrix(
                        self.raw_img, sigma=sigma, use_gaussian_derivatives=True
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

        self.raw_img_with_features = result

    @staticmethod
    def rescale_to_uint8(img):
        """
        Rescale the values of an array to the range (0, 255)
        """
        eps = 1e-6
        img_norm = (img - img.min()) / (img.max() - img.min() + eps)
        img_uint8 = (img_norm * 255).astype(np.uint8)
        return img_uint8

    @staticmethod
    def rescale_to_uint16(img):
        """
        Rescale the values of an array to the range (0, 65535)
        """
        eps = 1e-6
        img_norm = (img - img.min()) / (img.max() - img.min() + eps)
        img_uint16 = (img_norm * 65535).astype(np.uint16)
        return img_uint16
