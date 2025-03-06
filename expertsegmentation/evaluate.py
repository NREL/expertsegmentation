import os
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prettytable import PrettyTable
import seaborn as sns
from typing import Union

from expertsegmentation.data import SegDataset
from expertsegmentation.metrics import calculate_volume_fractions, calculate_kl_div, calculate_n_components_per_class


def convert_lambd_for_table(lambd):
    lambd_str = ''
    
    for elem in lambd.split(','):
        metric = elem.split(',')[0].split(':')[0].replace('\'', '').replace("{", '')
        val = elem.split(',')[0].split(':')[1].strip().replace('}', '')
        lambd_str += f"{val} ({metric})"
    
    return lambd_str


def plot(result_dict: dict,
         loss_dict: dict,
         n_epochs: int,
         img: np.ndarray,
         slice_idx_3d: int = None):
    """Display segmented image.

    Args:
        result_dict (dict): Dictionary returned from run_xgboost() with keys
                            "labels_default_loss" and "labels_custom_loss"
        loss_dict (dict):   Dictionary returned from run_xgboost()
        n_epochs (int):     Number of epochs model was trained for
        img (np.ndarray):   Original raw input image
        slice_idx_3d (int): If the data is 3D, provide a slice index to plot
                            (just xy plane for now)
    """

    yhat_default_loss = result_dict["labels_default_loss"]
    yhat_custom_loss = result_dict["labels_custom_loss"]

    _, ax = plt.subplots(nrows=3, ncols=len(yhat_custom_loss) + 2)
    if slice_idx_3d is None:
        ax[0, 0].imshow(img, cmap='gray')
    else:
        ax[0, 0].imshow(img[slice_idx_3d])
    ax[0, 0].set_title("Input image")
    if slice_idx_3d is None:
        ax[1, 0].imshow(yhat_default_loss)
    else:
        ax[1, 0].imshow(yhat_default_loss[slice_idx_3d])
    ax[1, 0].set_title("With native loss")

    # Plot raw output
    for i, lambd in enumerate(yhat_custom_loss.keys()):
        if isinstance(lambd, str):
            lambd_str = convert_lambd_for_table(lambd).split(" ")[0]
        else:
            lambd_str = lambd
        if slice_idx_3d is None:
            ax[0, i + 2].imshow(yhat_custom_loss[lambd])
        else:
            ax[0, i + 2].imshow(yhat_custom_loss[lambd][slice_idx_3d])
        ax[0, i + 2].set_title(f"Custom loss\nlambda={lambd_str}")
        ax[0, i+2].axis('off')

    # Plot difference maps
    for i, lambd in enumerate(yhat_custom_loss.keys()):
        diff = yhat_custom_loss[lambd] != yhat_default_loss
        if slice_idx_3d is None:
            ax[1, i + 2].imshow(diff, cmap="Reds")
        else:
            ax[1, i + 2].imshow(diff[slice_idx_3d], cmap="Reds")
        ax[1, i + 2].set_title("Difference map")
        ax[1, i+2].axis('off')

    # Plot loss curves
    loss_softmax = loss_dict["softmax_losses"]
    loss_custom = loss_dict["custom_losses"]
    for i, lambd in enumerate(loss_softmax.keys()):
        # n_epochs = len(loss_softmax[lambd])
        ax[2, i + 2].scatter(
            x=list(range(n_epochs)), y=loss_softmax[lambd], label="Softmax loss"
        )
        ax[2, i + 2].scatter(
            x=list(range(n_epochs)), y=loss_custom[lambd], label="Custom loss term"
        )
        ax[2, i + 2].scatter(
            x=list(range(n_epochs)),
            y=np.array(loss_custom[lambd]) + np.array(loss_softmax[lambd]),
            label="Total loss",
        )
        ax[2, i + 2].legend()
        ax[2, i + 2].set_xlabel("Epoch")
        ax[2, i + 2].set_ylabel("Loss")
        ax[2, i + 2].set_title("Loss curve")

    for i in range(3):
        ax[i, 0].axis('off')
        ax[i, 1].axis('off')
    return plt


def print_metrics_table(
                  result_dict: dict,
                  n_classes: int,
                  objectives: Union[list, str],
                  target_vf: dict):
    
    """Compute, display, and return dataframes of domain metrics.

    Args:
        result_dict: Dictionary returned by run_xgboost()
        n_classes: Number of phases in the image
        objectives: List or string of target objectives
        target_vf: Target volume fraction dictionary

    Return:
        metrics_df: Dataframe with volume fraction, number of components,
                    and average circularity for each class for prediction
                    with default loss and for each lambda with custom loss.

    """

    yhat_default_loss = result_dict["labels_default_loss"]
    yhat_custom_loss = result_dict["labels_custom_loss"]

    # Add metrics with default loss
    vfs_pred_default = calculate_volume_fractions(
        yhat_default_loss - 1, n_classes
    )
    n_components_pred_default = calculate_n_components_per_class(
        yhat_default_loss -1, n_classes
    )

    if "volume_fraction" in objectives:
        # Comparison to target
        kl_div = calculate_kl_div(
            np.array(list(vfs_pred_default.values())),
            target_vf,
        )

        evaluation_df = pd.DataFrame(
            {
                "lambda": ["N/A (default loss)"],
                "volume fraction: kl div to target": [kl_div],
            }
        )
    else:
        evaluation_df = pd.DataFrame()

    metrics_df = pd.DataFrame(
        {
            "lambda": ["N/A (default loss)"] * n_classes + [""],
            "class": list(range(1, n_classes + 1)) + [""],
            "volume fraction": list(vfs_pred_default.values()) + [""],
            "num_components": list(n_components_pred_default.values()) + [""],
        }
    )

    # Add a set of rows with custom loss for each value of lambda
    for lambd in yhat_custom_loss:
        vfs_pred = calculate_volume_fractions(
            yhat_custom_loss[lambd] - 1, n_classes
        )
        n_components_pred = calculate_n_components_per_class(
            yhat_custom_loss[lambd] - 1, n_classes
        )

        if "volume_fraction" in objectives:
            # Comparison to target
            kl_div = calculate_kl_div(
                np.array(list(vfs_pred.values())),
                target_vf,
            )

            temp_df = pd.DataFrame(
                {
                    "lambda": [convert_lambd_for_table(lambd)],
                    "volume fraction: kl div to target": [kl_div],
                }
            )
            evaluation_df = pd.concat([evaluation_df, temp_df])

        temp_df = pd.DataFrame(
            {
                "lambda": [convert_lambd_for_table(lambd)] * n_classes + [""],
                "class": list(range(1, n_classes + 1)) + [""],
                "volume fraction": list(vfs_pred.values()) + [""],
                "num_components": list(n_components_pred.values()) + [""],
            }
        )
        metrics_df = pd.concat([metrics_df, temp_df])

    # Prettyprint tables from dataframes
    tab = PrettyTable(list(metrics_df.columns))
    tab.add_rows(metrics_df.values.tolist())

    print(tab)

    if "volume_fraction" in objectives:
        tab_eval = PrettyTable(list(evaluation_df.columns))
        tab_eval.add_rows(evaluation_df.values.tolist())
        print("\n", tab_eval)

    return metrics_df, evaluation_df


def plot_steps_3d_slice(
                      result_dict: dict,
                      img: np.ndarray,
                      slice_idx: int = 0,
                      lambd: Union[int, float] = None,
                    ):    
    """
    Plot intermediate training steps.

    result_dict (dict): Dictionary returned by run_xgboost()
    img (np.ndarray):   The original image of interest.
    slice_idx (int):    Index of XY slice of volume to plot.
    lambd (float):      Weighting on target property.

    """
    step_dict = result_dict["intermediate_epoch_results"]

    # If a lambda is not provided, use the first one
    if lambd is None:
        lambd = next(iter(step_dict))
    if lambd not in step_dict:
        raise ValueError(f"lambd={lambd} not found in {step_dict.keys()}.")
    steps = step_dict[lambd].keys()

    fig, axes = plt.subplots(len(steps), 3)
    if len(axes.shape) == 1:
        axes = axes[None, :]
    axes[0, 0].set_title("Input image")
    axes[0, 1].set_title("Segmentation")
    axes[0, 2].set_title("Difference to\nsegmentation with\nnaive loss")
    for i, epoch in enumerate(steps):
        pred_labels = step_dict[lambd][epoch]['prediction'].argmax(axis=-1) + 1
        axes[i, 0].imshow(img[slice_idx], cmap='gray')
        axes[i, 1].imshow(pred_labels[slice_idx], cmap='gray_r')
        diff = pred_labels != result_dict['labels_default_loss']
        axes[i, 2].imshow(diff[slice_idx], cmap='Reds')
        for j in range(3):
            axes[i, j].tick_params(left = False, right = False , labelleft = False , 
                        labelbottom = False, bottom = False)
            axes[i, j].set_frame_on(False) 

    row_labels = ['Epoch {}'.format(epoch) for epoch in steps]
    for ax, row in zip(axes[:,0], row_labels):
        ax.set_ylabel(row, rotation='vertical')

    fig.tight_layout()
    return fig


def plot_loss_curve(loss_dict: dict,
                    lambd: float,
                    plot_type = 'all'):
    """
    Plot loss curves from training.

    Args:
        loss_dict (dict):   Dictionary returned from run_xgboost()
        lambd (float):      Weight on target property
        plot_type (str):    Reflects whether to plot all losses,
                            only the total, only the custom loss,
                            or only the softmax loss component.
    """
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


def save_gif(vol: np.ndarray, img_path: str):
    """Plot and save a gif of a 3D volume.

    Args:
        vol (np.ndarray): 3D volume to make gif of
        img_path (str): Path to directory to save gif

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
        result_dict (dict): Dictionary returned by run_xgboost.py
        dataset (SegDataset): Dataset used to segment the volume.

    """

    yhat_default_loss = result_dict["labels_default_loss"]
    yhat_custom_loss = result_dict["labels_custom_loss"]

    img_path = os.path.dirname(dataset.raw_img_fn)

    save_gif(yhat_default_loss, os.path.join(img_path, "result/default"))
    for lambd in yhat_custom_loss:
        save_gif(
            yhat_custom_loss[lambd], os.path.join(img_path, f"result/lambda={lambd}")
        )
