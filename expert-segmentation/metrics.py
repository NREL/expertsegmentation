"""
Define domain metrics of interest.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import entropy

from data import SegDataset, UserInputs


def calculate_volume_fractions(pred_labels: np.ndarray, n_classes: int):
    """
    Helper function to compute volume fractions per class.

    Assumes that the labels go from 0 to n_classes - 1.

    Args:
        pred_labels (np.ndarray)    Array with shape (n_pixels, ) which has values in
                                    {0, 1, 2, 3, 4} representing the predicted classes

    Return:
        result_dict (dict)          Dictionary with keys [0, 1, 2, 3, 4] and values that
                                    are the volume fractions per class, which sum to 1.

    """
    labels = list(range(n_classes))
    result_dict = dict()
    for label in labels:
        result_dict[label] = (pred_labels == label).sum() / pred_labels.size
    return result_dict


def calculate_kl_div(distr1: np.ndarray, distr2: np.ndarray):
    """
    Args:
        distr1: Array of probabilities that sum to 1
        distr2: Array of probabilities that sum to 1

    Return:
        kl_div: KL divergence between the two distributions
    """

    return entropy(distr1, distr2)


def calculate_perc_connected_components(labels: np.ndarray, n_classes: int, c: int):
    # TODO IMPLEMENT THIS IN 3D
    n_components_class_c, _ = cv2.connectedComponents((labels == c).astype(np.uint8))
    total_n_components = n_components_class_c
    for cl in list(range(n_classes)):
        if cl != c:
            n_components, _ = cv2.connectedComponents((labels == cl).astype(np.uint8))
            total_n_components += n_components
    return n_components_class_c / total_n_components


def calculate_connectivities(labels: np.ndarray, n_classes: int):
    classes = list(range(n_classes))
    result_dict = dict()
    for c in classes:
        result_dict[c] = 1 - calculate_perc_connected_components(labels, n_classes, c)
    return result_dict


def plot_results(labels: np.ndarray):
    if labels.ndim == 2:
        plt.figure()
        plt.imshow(labels)
        plt.title("Segmentation")
        plt.show()
    else:
        plt.figure()
        plt.imshow(labels[0])
        plt.title("Segmentation, slice 0")
        plt.show()


def print_metrics(labels: np.ndarray, dataset: SegDataset, user_input: UserInputs):
    vfs_pred = calculate_volume_fractions(labels - 1, dataset.n_classes)
    conns_pred = calculate_connectivities(labels - 1, dataset.n_classes)
    circs_pred = {c: np.nan for c in range(dataset.n_classes)}  # TODO

    kl_div = calculate_kl_div(
        np.array(list(vfs_pred.values())),
        np.array(list(user_input.volume_fraction_targets.values())),
    )

    metrics_df = pd.DataFrame(
        {
            "volume fraction": vfs_pred.values(),
            "connectivity": conns_pred.values(),
            "circularity": circs_pred.values(),
        },
        index=np.array(list(vfs_pred.keys())) + 1,
    )
    metrics_df.index.name = "class"

    print(metrics_df)
