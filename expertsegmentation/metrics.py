"""
Define domain metrics of interest.
"""

import numpy as np
import porespy as ps
from scipy.stats import entropy
from skimage.measure import label


def calculate_volume_fractions(pred_labels: np.ndarray, n_classes: int):
    """
    Compute volume fractions per class.

    Assumes that the labels count from 0 to n_classes - 1.

    Args:
        pred_labels (np.ndarray)    Array with shape (n_pixels, ) which has values in
                                    {0, 1, 2, 3, 4} representing the predicted classes

    Returns:
        result_dict (dict)          Dictionary with keys [0, 1, ..., n_classes-1] and values that
                                    are the volume fractions per class, which sum to 1.

    """
    labels = list(range(n_classes))
    result_dict = dict()
    for label in labels:
        result_dict[label] = (pred_labels == label).sum() / pred_labels.size
    return result_dict


def calculate_kl_div(distr1: np.ndarray, distr2: np.ndarray):
    """
    Calculate KL divergence between two distributions.

    Args:
        distr1 (np.ndarray): Array of probabilities that sum to 1
        distr2 (np.ndarray): Array of probabilities that sum to 1

    Return:
        kl_div (float): KL divergence between the two distributions
    """

    return entropy(distr1, distr2)


def calculate_n_components(labels: np.ndarray, target_class: int):
    """Calculate the number of isolated components that belong to
    the target class in a segmented image.

    Args:
        labels (np.ndarray):    Segmented image with integer type.
        target_class (int):     Label for the target class

    Return:
        n_isolated_components (int): Number of isolated, non-touching components
    """

    binary = (labels == target_class).astype(np.uint8)
    _, n_isolated_components = label(binary, return_num=True)
    return n_isolated_components


def calculate_n_components_per_class(labels: np.ndarray, n_classes: int):
    """Calculate number of isolated components per labeled class in an image.

    Args:
        labels (np.ndarray): Segmented image with integer type.
        n_classes (int): Number of labeled classes.

    Returns:
        (dict) Number of components per class.
    """
    result_dict = dict()
    for c in range(n_classes):
        result_dict[c] = calculate_n_components(labels, c)
    return result_dict


def calculate_average_circularity(labels: np.ndarray, c: int):
    """Calculate average circularity of class c in an image.

    Args:
        labels (np.ndarray): Segmented image with integer type.
        c (int): Class index of interest.

    Returns:
        (float) Average circularity of all isolated instances of class c
        in the image.
    """

    particles = (labels == c).astype(np.uint8)
    particles_uniquely_labeled, n = label(particles, return_num=True)

    # If the prediction is all 1 value, circularity has no meaning
    if n == 1:
        return -1

    props = ps.metrics.regionprops_3D(particles_uniquely_labeled)
    sphericities = [prop.sphericity for prop in props]
    return np.mean(sphericities)


def calculate_average_circularity_per_class(labels, n_classes):
    """Calculate average circularity per labeled class in an image.

    Args:
        labels (np.ndarray): Segmented image with integer type.
        n_classes (int): Number of labeled classes.

    Returns:
        (dict) Average circularity of all isolated instances of each class.
    """
    circs = dict()
    classes = list(range(n_classes))
    for c in classes:
        circs[c] = calculate_average_circularity(labels, c)
    return circs
