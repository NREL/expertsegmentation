from typing import Union

import numpy as np
import porespy as ps
from scipy.ndimage import distance_transform_edt
from skimage import measure
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb

from expertsegmentation.metrics import calculate_volume_fractions


def softmax(z: Union[float, np.ndarray]):
    """
    Softmax function

    Args:
        z (float, np.ndarray)   Input argument

    Returns:
        f(z) = exp(z) / sum(exp(z))
    """
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z), axis=1)).T
    return sm


def softmaxobj(preds: np.ndarray, dtrain: xgb.DMatrix):
    """Re-implementation of the native softmax loss in XGBoost.

    Args:
        preds: (N, K) array, N = #data, K = #classes.
        dtrain: DMatrix object with training data.

    Returns:
        grad: N*K array with gradient values.
        hess: N*K array with second-order gradient values.

    """
    # Label is a vector of class indices for each input example
    labels = dtrain.get_label()

    # When objective=softprob, preds has shape (N, K). Convert the labels
    # to one-hot encoding to match this shape.
    labels = OneHotEncoder(sparse_output=False).fit_transform(
        labels.reshape(-1, 1)
    )

    grad = preds - labels
    hess = 2.0 * preds * (1.0 - preds)

    # XGBoost wants them to be returned as 1-d vectors
    return grad.flatten(), hess.flatten()


def volume_fraction_obj(pred: np.ndarray, lambd: float, target_distr: np.ndarray):
    """

    Define volume fraction loss term to penalize when the volume
    fraction per class doesn't match the target volume fractions.

    loss =
        lambda * || [pred_vf_0, pred_vf_1, pred_vf_2, pred_vf_3, pred_vf_4]
                        - [vf_0, vf_1, vf_2, vf_3, vf_4] ||**2

    grad =
        2 * lambda * || [pred_vf_0, pred_vf_1, pred_vf_2, pred_vf_3, pred_vf_4]
                        - [vf_0, vf_1, vf_2, vf_3, vf_4] ||

    Args:
        pred (np.ndarray): Predicted probabilities per pixel (n_pixels, n_classes)
        lambd (float): Weighting on target volume fraction property.
        target_distr (np.ndarray):  Array with target volume fractions per phase.

    Returns:
        loss, gradient, hessian

    """

    pred_labels = np.argmax(pred, axis=1)
    pred_distr = np.array(
        list(calculate_volume_fractions(pred_labels, len(target_distr)).values())
    )
    loss = lambd * np.linalg.norm(pred_distr - target_distr) ** 2  # for the whole image
    grad_row = 2 * lambd * (pred_distr - target_distr)
    grad = np.array([grad_row] * len(pred))
    return loss, grad.flatten(), 0


def connectivity_obj(
    pred: np.ndarray, lambd: int, c: int, n_classes: int, H: int, W: int, D: int = None, direction: str = "max",
):
    """
    
    If minimizing connectivity:
        Loss (L) = lambd * 1 / number_of_components
        i.e. loss is minimized when there are a greater number of isolated components
    
    If maximizing connectivity:
        Loss (L) = lambd * log(number_of_components)
        i.e. loss is minimized when there are a smaller number of isolated components

    Args:
        pred: Predicted probabilities per pixel (n_pixels, n_classes)
        lambd: Lambda to weight connectivity loss term
        c: The class of interest
        n_classes: Number of unique labeled classes in the dataset
        H: Height of original input image
        W: Width of original input image
        D: Depth of original input image, if 3D, else None
        direction: One of {"max", "min"}. Whether to maximize or minimize connectivity
                   of the target phase.
    
    """
    if D is None:
        pred_labels = np.argmax(pred, axis=1).reshape((H, W)).astype(np.uint8)
    else:
        pred_labels = np.argmax(pred, axis=1).reshape((H, W, D)).astype(np.uint8)

    # Binarize the image
    binary = (pred_labels == c).astype(np.uint8)

    # Get the number of components
    particles_uniquely_labeled, n_isolated_components = measure.label(binary, return_num=True)

    # If the prediction is all one value, distance map doesn't make sense.
    # Defer to softmax loss and set this loss term to 0
    if len(np.unique(binary)) <= 1:
        loss = 0
        gradient = np.zeros((*pred_labels.shape, n_classes))

    else:

        # Loss
        if direction == "min":
            # Min connectivity = maximize number of isolated components
            loss = lambd * 1 / n_isolated_components

            # Gradient
            distance_map = distance_transform_edt(binary)
            # Division by 0 --> 0
            distance_map_inv = np.divide(
                1, distance_map, out=np.zeros(distance_map.shape), where=distance_map != 0
            )

            # Gradient is largest at particle boundaries and smallest within particles
            gradient = lambd * distance_map_inv * loss
        
        else:
            # Max connectivity = minimize number of isolated components
            loss = lambd * np.log(n_isolated_components)

            ## Gradient term 1: towards deletion of noisy non-target pixels ##
            # (Normalized) particle area of each component
            area_map = particles_uniquely_labeled
            props = ps.metrics.regionprops_3D(particles_uniquely_labeled)
            areas = [prop.area for prop in props]
            max_area = np.max(areas)
            for prop in props:
                area_map[area_map == prop.label] = prop.area / max_area
            gradient_area = lambd * area_map * loss


            ## Gradient term 2: towards growth of target pixels at boundaries ##
            # Flip the binarized image so that the distance map
            # is the distance from non-particle to particle
            binary_inverse = 1 - binary
            
            # Distance map is largest far from particles, smallest
            # near particle boundaries, and 0 inside particles
            distance_map = distance_transform_edt(binary_inverse)

            # Invert the distance map to be largest close to particles,
            # smallest far from particles, and 0 inside particles.
            # Division by 0 --> 0
            distance_map_inv = np.divide(
                1, distance_map, out=np.zeros(distance_map.shape), where=distance_map != 0
            )
            gradient_distancemap = lambd * distance_map_inv * loss

            # Combine the two gradient terms
            gradient = gradient_area + gradient_distancemap


        # Per-class gradient
        gradient = -1 * np.stack([gradient] * n_classes)
        gradient[c] *= -1
        gradient = np.moveaxis(gradient, 0, -1)

    return loss, gradient.flatten(), 0
