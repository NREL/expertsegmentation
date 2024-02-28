"""
Implementation of XGBoost with custom loss.
"""

import GPUtil
import numpy as np
import porespy as ps
from scipy.ndimage import distance_transform_edt
from skimage import measure
from sklearn.preprocessing import OneHotEncoder
from torch import from_numpy
from torch.nn import CrossEntropyLoss
import xgboost as xgb

from data import SegDataset, UserInputs
from metrics import (
    calculate_volume_fractions,
    calculate_circularity,
    calculate_kl_div,
)


def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z), axis=1)).T
    return sm


def softmaxobj(preds, dtrain):
    """Re-implementation of the native softmax loss in XGBoost.

    Args:
        preds: (N, K) array, N = #data, K = #classes.
        dtrain: DMatrix object with training data.
        h: Height of image for reshaping
        w: Width of image for reshaping

    Returns:
        grad: N*K array with gradient values.
        hess: N*K array with second-order gradient values.

    """
    # Label is a vector of class indices for each input example
    labels = dtrain.get_label()

    # When objective=softprob, preds has shape (N, K). Convert the labels
    # to one-hot encoding to match this shape.
    labels = OneHotEncoder(sparse="deprecated", sparse_output=False).fit_transform(
        labels.reshape(-1, 1)
    )

    grad = preds - labels
    hess = 2.0 * preds * (1.0 - preds)

    # XGBoost wants them to be returned as 1-d vectors
    return grad.flatten(), hess.flatten()


def volume_fraction_obj(pred, lambd, target_distr):
    """

    Define volume fraction loss term to penalize when the volume
    fraction per class doesn't match the target volume fractions.

    loss =
        lambda * || [pred_vf_0, pred_vf_1, pred_vf_2, pred_vf_3, pred_vf_4]
                        - [vf_0, vf_1, vf_2, vf_3, vf_4] ||**2

    grad =
        2 * lambda * || [pred_vf_0, pred_vf_1, pred_vf_2, pred_vf_3, pred_vf_4]
                        - [vf_0, vf_1, vf_2, vf_3, vf_4] ||

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


def circularity_obj(
    pred: np.ndarray,
    lambd: int,
    c: int,
    target_circularity: float,
    img: np.ndarray,
    H: int,
    W: int,
    D: int = None,
):
    """
    Define "circularity" loss term to penalize the shape of a particular
    class towards a target average circularity across the image.

    For every connected component of class c in the image:

        circularity = feret max diameter / equivalent circle area diameter

    where the feret max diameter is the longest distance between points
    around a region’s convex hull contour, and equivalent circle area diameter
    is the diameter of a circle with the same area as the region.

    loss =
        lambda * || predicted_average_circularity - target_circularity ||**2

    gradient =
        2 * lambda * || predicted_average_circularity - target_circularity ||


    Args:
        pred: Predicted probabilities per pixel (n_pixels, n_classes)
        lambd: Lambda to weight connectivity loss term
        c: The class of interest
        target_ciricularity: Average circularity
        img: The original image with intensity values
        H: Height of original input image
        W: Width of original input image
        D: Depth of original input image, if 3D, else None

    """
    if D is None:
        pred_labels = np.argmax(pred, axis=1).reshape((H, W)).astype(np.uint8)
    else:
        pred_labels = np.argmax(pred, axis=1).reshape((H, W, D)).astype(np.uint8)

    circularity = calculate_circularity(pred_labels, img, c)
    loss = lambd * np.linalg.norm(circularity - target_circularity) ** 2
    grad_val = 2 * lambd * circularity
    gradient = np.full(pred.shape, grad_val)

    return loss, gradient.flatten(), 0


def run_xgboost(dataset: SegDataset, user_input: UserInputs, save_steps: list[int] = [5, 25, 50, 75, 95]):
    """Main function to fit XGBoost model and make predictions.

    Args:
        dataset: Object with raw input and labeled image.
        user_input: UserInputs argument specifying targets.
        save_steps: List of epoch integers to save intermediate output for.

    Return:
        pred: Predictions on input image.
    """
    if GPUtil.getAvailable():
        device = "cuda"
        print("GPU is available. Using GPU.")
    else:
        device = "cpu"
        print("GPU unavailable. Using cpu.")
    params = {
        "max_depth": 2,
        "eta": 0.1,
        "device": device,
        "objective": "multi:softprob",
        "num_class": dataset.n_classes,
    }

    # The unlabeled pixels are 0 in Ilastik. For XGBoost, the class indices
    # should start from 0, so redefine "unlabeled" to be the largest index.
    dataset.labeled_img[dataset.labeled_img == 0] = dataset.n_classes + 1
    dataset.labeled_img = dataset.labeled_img - 1
    unlabeled = dataset.n_classes

    # Select the labeled pixels for training
    X_train = dataset.raw_img_with_features[dataset.labeled_img != unlabeled]
    y_train = dataset.labeled_img[dataset.labeled_img != unlabeled]

    # Reshape the image into feature matrices with shape (H*W, C) or (H*W*D, C)
    X_test = dataset.raw_img_with_features.reshape(
        -1, dataset.raw_img_with_features.shape[-1]
    )

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtrain_full_image = xgb.DMatrix(X_test)
    dtest = dtrain_full_image  # xgb.DMatrix(X_test)

    # Run default loss first
    print("Evaluating with native loss...")
    model_default_loss = xgb.train(params, dtrain, 100)
    yhat_probs_default = model_default_loss.predict(dtest)
    yhat_probs_default = yhat_probs_default.reshape(
        (*dataset.raw_img_with_features.shape[:-1], dataset.n_classes)
    )
    # Add 1 to map back to original labels
    yhat_labels_default = np.argmax(yhat_probs_default, axis=-1) + 1

    # Instantiate the model
    model = xgb.Booster(params, [dtrain])

    # Define default loss
    loss = CrossEntropyLoss()

    # Run model for each lambda
    lambdas = user_input.lambdas
    softmax_losses_per_lambda = dict()
    custom_losses_per_lambda = dict()
    yhat_probabilities_per_lambda = dict()
    yhat_labels_per_lambda = dict()
    step_dict = dict()

    for l in range(user_input.n_lambdas):

        print("Evaluating for lambdas: {}".format({key: lambdas[key][l] for key in user_input.objective}))

        # Run training and save losses (100 epochs)
        softmax_losses = []
        custom_losses = []
        step_predictions = dict()
        for i in range(100):
            if i % 10 == 0:
                print(f"\tepoch {i}")
            pred = model.predict(dtrain)

            # Softmax loss ONLY FOR THE TRAINING PIXELS (the ones w a label)
            g_softmax, h_softmax = softmaxobj(pred, dtrain)

            # Custom loss across the entire image
            pred_full_image = model.predict(dtrain_full_image)

            l_custom_sum = 0
            g_custom_sum = []
            h_custom_sum = 0
            for objective in user_input.objective:
                if objective == "volume_fraction":
                    l_custom, g_custom, h_custom = volume_fraction_obj(
                        pred_full_image,
                        lambdas["volume_fraction"][l],
                        target_distr=user_input.volume_fraction_targets,
                    )
                elif objective == "connectivity":
                    l_custom, g_custom, h_custom = connectivity_obj(
                        pred_full_image,
                        lambdas["connectivity"][l],
                        user_input.connectivity_target - 1,
                        dataset.n_classes,
                        *dataset.raw_img.shape,
                        user_input.connectivity_direction,
                    )
                elif objective == "circularity":
                    l_custom, g_custom, h_custom = circularity_obj(
                        pred_full_image,
                        lambdas["circularity"][l],
                        user_input.circularity_target_class - 1,
                        user_input.circularity_target_value,
                        dataset.raw_img,
                        *dataset.raw_img.shape,
                    )
                else:
                    raise ValueError(
                        f"Objective should be one of {'volume_fraction', 'connectivity', 'circularity'}. Received {user_input.objective}"
                    )
                
                l_custom_sum += l_custom
                g_custom_sum.append(g_custom)
                h_custom_sum += h_custom

            # Aggregate
            g_custom_sum = np.sum(g_custom_sum, axis=0)

            # Re-reduce to only the labeled pixels
            g_custom_before_reshape = g_custom
            g_custom = g_custom.reshape(
                (*dataset.raw_img_with_features.shape[:-1], pred.shape[-1])
            )[dataset.labeled_img != unlabeled].flatten()

            # Compute the actual softmax loss and save for later
            l_softmax = loss(
                from_numpy(pred), from_numpy(dtrain.get_label().astype(np.uint8))
            )
            softmax_losses.append(l_softmax)
            custom_losses.append(l_custom)

            g = g_softmax + g_custom
            h = h_softmax + h_custom

            model.boost(dtrain, g, h)

            # Save intermediate steps
            if i in save_steps:
                step_predictions[i] = {'prediction': pred_full_image.reshape((*dataset.raw_img_with_features.shape[:-1], dataset.n_classes)),
                                       'l_softmax': l_softmax,
                                       'l_custom': l_custom,
                                       'g_softmax': g_softmax,
                                       'g_custom': g_custom_before_reshape,
                }

        # Evaluate
        yhat_probs = model.predict(dtest).reshape(
            (*dataset.raw_img_with_features.shape[:-1], dataset.n_classes)
        )
        yhat_labels = np.argmax(yhat_probs, axis=-1)
        # Reset to original labels
        yhat_labels = yhat_labels + 1

        lambd_str = "{}".format({key: lambdas[key][l] for key in user_input.objective})
        softmax_losses_per_lambda[lambd_str] = softmax_losses
        custom_losses_per_lambda[lambd_str] = custom_losses
        yhat_probabilities_per_lambda[lambd_str] = yhat_probs
        yhat_labels_per_lambda[lambd_str] = yhat_labels
        step_dict[lambd_str] = step_predictions

    loss_dict = {
        "softmax_losses_only": softmax_losses_per_lambda,
        "custom_losses_only": custom_losses_per_lambda,
    }
    result_dict = {
        "probabilities_default_loss": yhat_probs_default,
        "labels_default_loss": yhat_labels_default,
        "probabilities_custom_loss": yhat_probabilities_per_lambda,
        "labels_custom_loss": yhat_labels_per_lambda,
    }

    return (
        loss_dict,
        result_dict,
        step_dict,
    )
