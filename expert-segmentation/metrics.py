"""
Define domain metrics of interest.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
from scipy.stats import entropy
from skimage.measure import regionprops_table

from data import SegDataset, UserInputs


def calculate_circularity(labels: np.ndarray, image: np.ndarray, c: int):
    """Calculate average circularity of given class in an image.

    Args:
        labels: Segmented image with integer type.
        image: The original raw image.
        c: Class index of interest.

    Return:
        Average circularity of all isolated instances of class c
        in the image.
    """

    particles = (labels == c).astype(np.uint8)
    n, particles_uniquely_labeled = cv2.connectedComponents(particles)

    # If the prediction is all 1 value, circularity has no meaning
    if n == 1:
        return -1

    props = regionprops_table(
        particles_uniquely_labeled,
        intensity_image=image,
        properties=(
            "area",
            "perimeter",
            "major_axis_length",
            "minor_axis_length",
        ),
    )

    small_particle = props["perimeter"] == 0
    circularity = (
        4
        * np.pi
        * props["area"][~small_particle]
        / props["perimeter"][~small_particle] ** 2
    )

    return circularity.mean()


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


def calculate_connectivities(labels: np.ndarray, n_classes: int):
    """Calculate connectivity of a segmented image for each unique class.

    Define connectivity as (1 - normalized number of isolated components).
    Assumes that the labels go from 0 to n_classes - 1.

    Args:
        labels:     Segmented image with integer type.
        n_classes:  Number of unique classes in the dataset

    Return:
        result_dict:    Dictionary with connectivity value per class.
    """
    classes = list(range(n_classes))

    # Array of number of isolated components per class
    n_isolated_components = np.zeros(n_classes)
    for c in classes:
        n_isolated_components[c] = cv2.connectedComponents(
            (labels == c).astype(np.uint8)
        )[0]
    conn_dict = {
        c: 1 - n_isolated_components[c] / n_isolated_components.sum() for c in classes
    }
    return conn_dict


def plot_results(result_dict: dict, loss_dict: dict, slice_idx_3d: int = None):
    """Display segmented image.

    Args:
        result_dict:    Dictionary returned from run_xgboost() with keys
                        "labels_default_loss" and "labels_custom_loss"
        slice_idx_3d:   If the data is 3D, provide a slice index to plot
                        (just xy plane for now)
    """

    yhat_default_loss = result_dict["labels_default_loss"]
    yhat_custom_loss = result_dict["labels_custom_loss"]

    _, ax = plt.subplots(nrows=3, ncols=len(yhat_custom_loss) + 1)
    if slice_idx_3d is None:
        ax[0, 0].imshow(yhat_default_loss)
    else:
        ax[0, 0].imshow(yhat_default_loss[slice_idx_3d])
    ax[0, 0].set_title("With native loss")

    # Plot raw output
    for i, lambd in enumerate(yhat_custom_loss.keys()):
        if slice_idx_3d is None:
            ax[0, i + 1].imshow(yhat_custom_loss[lambd])
        else:
            ax[0, i + 1].imshow(yhat_custom_loss[lambd][slice_idx_3d])
        ax[0, i + 1].set_title(f"Custom loss, lambda={lambd}")

    # Plot difference maps
    ax[1, 0].axis("off")
    for i, lambd in enumerate(yhat_custom_loss.keys()):
        diff = yhat_custom_loss[lambd] != yhat_default_loss
        if slice_idx_3d is None:
            ax[1, i + 1].imshow(diff, cmap="Reds")
        else:
            ax[1, i + 1].imshow(diff[slice_idx_3d], cmap="Reds")
        ax[1, i + 1].set_title("Difference map")

    # Plot loss curves
    loss_softmax = loss_dict["softmax_losses_only"]
    loss_custom = loss_dict["custom_losses_only"]
    ax[2, 0].axis("off")
    for i, lambd in enumerate(loss_softmax.keys()):
        ax[2, i + 1].scatter(
            x=list(range(100)), y=loss_softmax[lambd], label="Softmax loss"
        )
        ax[2, i + 1].scatter(
            x=list(range(100)), y=loss_custom[lambd], label="Custom loss term"
        )
        ax[2, i + 1].scatter(
            x=list(range(100)),
            y=np.array(loss_custom[lambd]) + np.array(loss_softmax[lambd]),
            label="Total loss",
        )
        ax[2, i + 1].legend()
        ax[2, i + 1].set_xlabel("Epoch")
        ax[2, i + 1].set_ylabel("Loss")
        ax[2, i + 1].set_title("Loss curve")

    plt.show()


def print_metrics(result_dict: dict, dataset: SegDataset, user_input: UserInputs):

    yhat_default_loss = result_dict["labels_default_loss"]
    yhat_custom_loss = result_dict["labels_custom_loss"]

    # Initialize the table
    tab = PrettyTable(
        ["Lambda", "Class", "Volume Fraction", "Connectivity", "Circularity"]
    )

    # Add metrics with default loss
    vfs_pred = calculate_volume_fractions(yhat_default_loss - 1, dataset.n_classes)
    conns_pred = calculate_connectivities(yhat_default_loss - 1, dataset.n_classes)
    circs_pred = calculate_circularity(yhat_default_loss - 1, dataset.n_classes)
    kl_div = calculate_kl_div(
        np.array(list(vfs_pred.values())),
        user_input.volume_fraction_targets,
    )
    tab.add_rows(
        [
            ("N/A (Default Loss)", i + 1, vfs_pred[i], conns_pred[i], circs_pred[i])
            for i in range(dataset.n_classes)
        ]
    )
    tab.add_row(["", "", f"KL div to target: {kl_div}", "", ""], divider=True)

    # Add a set of rows with custom loss for each value of lambda
    for lambd in yhat_custom_loss:
        vfs_pred = calculate_volume_fractions(
            yhat_custom_loss[lambd] - 1, dataset.n_classes
        )
        conns_pred = calculate_connectivities(
            yhat_custom_loss[lambd] - 1, dataset.n_classes
        )
        circs_pred = calculate_circularity(
            yhat_custom_loss[lambd] - 1, dataset.n_classes
        )

        kl_div = calculate_kl_div(
            np.array(list(vfs_pred.values())),
            user_input.volume_fraction_targets,
        )

        tab.add_rows(
            [
                (lambd, i + 1, vfs_pred[i], conns_pred[i], circs_pred[i])
                for i in range(dataset.n_classes)
            ]
        )
        tab.add_row(["", "", f"KL div to target: {kl_div}", "", ""], divider=True)

    print(tab)
