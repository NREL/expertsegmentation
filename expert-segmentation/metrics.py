"""
Define domain metrics of interest.
"""

import os
import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from scipy.stats import entropy
import seaborn as sns
from skimage.measure import label, regionprops_table
from skimage.morphology import remove_small_objects


from data import SegDataset, UserInputs


def calculate_circularity(labels: np.ndarray, c: int):
    """Calculate average circularity of given class in an image.

    Args:
        labels: Segmented image with integer type.
        c: Class index of interest.

    Return:
        Average circularity of all isolated instances of class c
        in the image.
    """

    particles = (labels == c).astype(np.uint8)
    particles_uniquely_labeled, n = label(particles, return_num=True)

    # If the prediction is all 1 value, circularity has no meaning
    if n == 1:
        return -1

    # Remove tiny particles - this messes up the feret diameter calculation
    particles_uniquely_labeled = remove_small_objects(
        particles_uniquely_labeled, min_size=8
    )

    props = regionprops_table(
        particles_uniquely_labeled,
        properties=(
            "equivalent_diameter_area",
            "feret_diameter_max",
        ),
    )

    circularity = props["feret_diameter_max"] / props["equivalent_diameter_area"]

    return circularity.mean()


def calculate_circularities(labels, n_classes):
    circs = dict()
    classes = list(range(n_classes))
    for c in classes:
        circs[c] = calculate_circularity(labels, c)
    return circs


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


def calculate_n_components(labels: np.ndarray, target_class: int):
    """Calculate connectivity of a segmented image for each unique class.

    Define connectivity as (1 - normalized number of isolated components).
    Assumes that the labels go from 0 to n_classes - 1.

    Args:
        labels:        Segmented image with integer type.
        target_class:  Label for the target class

    Return:
        n_isolated_components: Number of isolated, non-touching components
    """

    binary = (labels == target_class).astype(np.uint8)
    _, n_isolated_components = label(binary, return_num=True)
    return n_isolated_components


def save_gif(vol: np.ndarray, img_path: str):
    """Plot and save a gif of a 3D volume.

    Args:
        vol: 3D volume to make gif of
        save_dir: Path to directory to save gif

    """

    if vol.ndim != 3:
        raise ValueError(
            f"Expected 3D volume. Received volume with dimension {vol.ndim}"
        )

    dim = ["x", "y", "z"]

    os.makedirs(img_path)

    # Save a png of each slice
    for d in range(3):
        img_path_d = "/".join([img_path, "dim_{}".format(dim[d])])
        print("img_path_d: ", img_path_d)
        if not os.path.exists(img_path_d):
            os.mkdir(img_path_d)
        for i in range(vol.shape[d]):
            plt.figure()
            if d == 0:
                img = vol[i, :, :]
            elif d == 1:
                img = vol[:, i, :]
            elif d == 2:
                img = vol[:, :, i]
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
            plt.savefig(
                img_path_d + "/{:03d}.png".format(i), dpi=150, bbox_inches="tight"
            )
            plt.close()

    for d in range(3):
        img_path_d = "/".join([img_path, "dim_{}".format(dim[d])])
        images = []
        for fn in os.listdir(img_path_d):
            img = imageio.imread(os.path.join(img_path_d, fn))
            images.append(img)
        imageio.mimsave(os.path.join(img_path, f"dim_{dim[d]}.gif"), images)


def save_gifs(result_dict: dict, dataset: SegDataset):
    """

    Saves gifs in each direction in folder called 'results' in the same
    directory where the original input image is. Separate folder within
    'results' for default loss and for each value of lambda run.

    Args:
        result_dict: Dictionary returned by run_xgboost.py
        dataset: Dataset used to segment the volume.

    """

    yhat_default_loss = result_dict["labels_default_loss"]
    yhat_custom_loss = result_dict["labels_custom_loss"]

    img_path = os.path.dirname(dataset.raw_img_fn)

    save_gif(yhat_default_loss, os.path.join(img_path, "result/default"))
    for lambd in yhat_custom_loss:
        save_gif(
            yhat_custom_loss[lambd], os.path.join(img_path, f"result/lambda={lambd}")
        )


def plot_loss_curve(loss_dict, lambd, plot_type = 'all'):

    plot_types = {'all', 'total_only', 'custom_only', 'softmax_only'}
    if plot_type not in plot_types:
        raise ValueError(f"`plot_type` must be one of {plot_types}. Received {plot_type}.")


    loss_list = np.array(loss_dict['custom_losses_only'][lambd]) + np.array(loss_dict['softmax_losses_only'][lambd])
    sns.set_context("paper", font_scale = 2)
    fig, ax = plt.subplots(figsize=(10, 6))

    if plot_type == 'total_only':
        sns.lineplot(x = list(range(len(loss_list))),
                    y = loss_list,
                    ax = ax,
                    marker='o')
        ax.set(ylabel='Total loss')

    elif plot_type == 'softmax_only':
        sns.lineplot(x = list(range(len(loss_list))),
            y = np.array(loss_dict['softmax_losses_only'][lambd]),
            ax = ax,
            marker='o')
        ax.set(ylabel='Softmax loss')

    elif plot_type == 'custom_only':
        sns.lineplot(x = list(range(len(loss_list))),
            y = np.array(loss_dict['custom_losses_only'][lambd]),
            ax = ax,
            marker='o')
        ax.set(ylabel='Custom loss')

    elif plot_type == 'all':
        sns.lineplot(x = list(range(len(loss_list))),
            y = loss_list,
            ax = ax,
            marker='o')
        sns.lineplot(x = list(range(len(loss_list))),
            y = np.array(loss_dict['custom_losses_only'][lambd]),
            ax = ax,
            marker='o')
        sns.lineplot(x = list(range(len(loss_list))),
            y = np.array(loss_dict['softmax_losses_only'][lambd]),
            ax = ax,
            marker='o')
        ax.legend(['Total loss', 'Volume fraction loss', 'Softmax loss'])
        ax.set(ylabel='Loss')
    ax.set(xlabel='Epochs')
    plt.show()


def plot_steps_3d_slice(step_dict: dict,
                      result_dict: dict,
                      lambd: float,
                      dataset,
                      steps: list[int] = None,
                      slice_idx: int = 0,
                    ):
    
    if steps is None:
        steps = list(step_dict.keys())

    fig, axes = plt.subplots(len(steps), 3)
    for i, epoch in enumerate(steps):
        pred_labels = step_dict[lambd][epoch]['prediction'].argmax(axis=-1) + 1
        axes[i, 0].imshow(dataset.raw_img[slice_idx], cmap='gray')
        axes[i, 1].imshow(pred_labels[slice_idx], cmap='gray_r')
        diff = pred_labels != result_dict['labels_default_loss']
        axes[i, 2].imshow(diff[slice_idx], cmap='Reds')
        for j in range(3):
            axes[i, j].tick_params(left = False, right = False , labelleft = False , 
                        labelbottom = False, bottom = False)
            axes[i, j].set_frame_on(False) 

    row_labels = ['Epoch {}'.format(epoch) for epoch in steps]
    for ax, row in zip(axes[:,0], row_labels):
        ax.set_ylabel(row, rotation='vertical', size='xx-small')

    fig.tight_layout()
    plt.subplots_adjust(wspace=0)
    plt.show()


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
    """Compute, display, and return dataframes of domain metrics.

    Args:
        result_dict: Dictionary returned by run_xgboost()
        dataset: Original dataset used for the segmentation
        user_input: User input dictionary with domain knowledge targets

    Return:
        metrics_df: Dataframe with volume fraction, connectivity, and circularity
                    for each class for prediction with default loss and for each
                    lambda with custom loss.
        evaluation_df: Dataframe with comparison to target for prediction with default
                       loss and for each lambda with custom loss.

    """

    yhat_default_loss = result_dict["labels_default_loss"]
    yhat_custom_loss = result_dict["labels_custom_loss"]

    # Add metrics with default loss
    vfs_pred_default = calculate_volume_fractions(
        yhat_default_loss - 1, dataset.n_classes
    )
    circs_pred_default = calculate_circularities(
        yhat_default_loss - 1, dataset.n_classes
    )
    kl_div = calculate_kl_div(
        np.array(list(vfs_pred_default.values())),
        user_input.volume_fraction_targets,
    )
    circ_perc_change = (
        circs_pred_default[user_input.circularity_target_class - 1]
        - user_input.circularity_target_value
    ) / user_input.circularity_target_value

    metrics_df = pd.DataFrame(
        {
            "lambda": ["N/A (default loss)"] * dataset.n_classes + [""],
            "class": list(range(1, dataset.n_classes + 1)) + [""],
            "volume fraction": list(vfs_pred_default.values()) + [""],
            "circularity": list(circs_pred_default.values()) + [""],
        }
    )

    evaluation_df = pd.DataFrame(
        {
            "lambda": ["N/A (default loss)"],
            "volume fraction: kl div to target": [kl_div],
            "circularity: % change from target": [circ_perc_change],
        }
    )

    # Add a set of rows with custom loss for each value of lambda
    for lambd in yhat_custom_loss:
        vfs_pred = calculate_volume_fractions(
            yhat_custom_loss[lambd] - 1, dataset.n_classes
        )
        circs_pred = calculate_circularities(
            yhat_custom_loss[lambd] - 1, dataset.n_classes
        )

        kl_div = calculate_kl_div(
            np.array(list(vfs_pred.values())),
            user_input.volume_fraction_targets,
        )
        circ_perc_change = (
            circs_pred[user_input.circularity_target_class - 1]
            - user_input.circularity_target_value
        ) / user_input.circularity_target_value

        temp_df = pd.DataFrame(
            {
                "lambda": [lambd] * dataset.n_classes + [""],
                "class": list(range(1, dataset.n_classes + 1)) + [""],
                "volume fraction": list(vfs_pred.values()) + [""],
                "circularity": list(circs_pred.values()) + [""],
            }
        )
        metrics_df = pd.concat([metrics_df, temp_df])

        temp_df = pd.DataFrame(
            {
                "lambda": [lambd],
                "volume fraction: kl div to target": [kl_div],
                "circularity: % change from target": [circ_perc_change],
            }
        )
        evaluation_df = pd.concat([evaluation_df, temp_df])

    # Prettyprint tables from dataframes
    tab = PrettyTable(list(metrics_df.columns))
    tab.add_rows(metrics_df.values.tolist())
    tab_eval = PrettyTable(list(evaluation_df.columns))
    tab_eval.add_rows(evaluation_df.values.tolist())
    print(tab, "\n", tab_eval)

    return metrics_df, evaluation_df
