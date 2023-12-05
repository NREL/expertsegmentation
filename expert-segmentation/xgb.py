"""
Implementation of XGBoost with custom loss.
"""

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from torch import from_numpy
from torch.nn import CrossEntropyLoss
import xgboost as xgb

from data import SegDataset, UserInputs
from metrics import (
    calculate_volume_fractions,
    calculate_connectivities,
    calculate_circularity,
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
    pred: np.ndarray, lambd: int, c: int, n_classes: int, H: int, W: int, D: int = None
):
    """
    Define "connectivity" loss term to penalize the connectivity of
    a particular class. i.e. the "blue" class (carbon binder) should
    be highly connected.

    Args:
        pred: Predicted probabilities per pixel (n_pixels, n_classes)
        lambd: Lambda to weight connectivity loss term
        c: The class of interest
        n_classes: Number of unique labeled classes in the dataset
        H: Height of original input image
        W: Width of original input image
        D: Depth of original input image, if 3D, else None

    loss =
        lambda * || 1 - n_components in class 0 / total n_components || **2

    gradient =
        2 * lambda * || 1 - n_components in class 0 / total n_components ||

    """
    if D is None:
        pred_labels = np.argmax(pred, axis=1).reshape((H, W)).astype(np.uint8)
    else:
        pred_labels = np.argmax(pred, axis=1).reshape((H, W, D)).astype(np.uint8)

    connectivity = calculate_connectivities(pred_labels, n_classes=n_classes)[c]
    loss = (
        -1 * lambd * connectivity**2
    )  # Negative because want to *maximize* connectivity
    grad_val = -1 * 2 * lambd * connectivity
    gradient = np.full(pred.shape, grad_val)

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

        circularity = 4 * pi * Area / Perimeter**2

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


def run_xgboost(dataset: SegDataset, user_input: UserInputs):
    """Main function to fit XGBoost model and make predictions.

    Args:
        dataset: Object with raw input and labeled image.

    Return:
        pred: Predictions on input image.
    """
    params = {
        "max_depth": 2,
        "eta": 0.1,
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
    yhat_labels_default = (
        np.argmax(yhat_probs_default, axis=-1) + 1
    )  # Adding 1 to map back to original labels

    # Instantiate the model
    model = xgb.Booster(params, [dtrain])

    # Define default loss
    loss = CrossEntropyLoss()

    # Get user inputs (for now one type of input at a time)
    # volume_fraction_targets = user_input.volume_fraction_targets
    # connectivity_target = (
    # user_input.connectivity_target - 1
    # )  # subtract to map to xgboost labels
    circularity_target_class = (
        user_input.circularity_target_class - 1
    )  # Subtract to map to xgboost labels
    circularity_target_value = user_input.circularity_target_value

    # Run model for each lambda
    softmax_losses_per_lambda = dict()
    custom_losses_per_lambda = dict()
    yhat_probabilities_per_lambda = dict()
    yhat_labels_per_lambda = dict()
    for lambd in user_input.lambdas:

        print(f"Evaluating for lambda = {lambd}...")

        # Run training and save losses (100 epochs)
        softmax_losses = []
        custom_losses = []
        for i in range(100):
            if (i%10 == 0):
                print("\tepoch {i}")
            pred = model.predict(dtrain)
            g_softmax, h_softmax = softmaxobj(
                pred, dtrain
            )  # ONLY FOR THE TRAINING PIXELS (the ones w a label)

            pred_full_image = model.predict(dtrain_full_image)  # ENTIRE IMAGE
            # l_vf, g_vf, h_vf = volume_fraction_obj(
            #     pred_full_image, lambd=lambd, target_distr=volume_fraction_targets
            # )  # VOLUME FRACTION PENALTY ON ENTIRE IMAGE
            # l_conn, g_conn, h_conn = connectivity_obj(
                # pred_full_image,
                # lambd,
                # connectivity_target,
                # dataset.n_classes,
                # *dataset.raw_img.shape,
            # )  # CONNECTIVITY PENALTY ON THE ENTIRE IMAGE
            l_circ, g_circ, h_circ = circularity_obj(
                pred_full_image,
                lambd,
                circularity_target_class,
                circularity_target_value,
                dataset.raw_img,
                *dataset.raw_img.shape,
            )

            # Re-reduce to only the labeled pixels
            # g_vf = g_vf.reshape(
            #     (*dataset.raw_img_with_features.shape[:-1], pred.shape[-1])
            # )[dataset.labeled_img != unlabeled].flatten()
            # g_conn = g_conn.reshape(
            #     (*dataset.raw_img_with_features.shape[:-1], pred.shape[-1])
            # )[
            #     dataset.labeled_img != unlabeled
            # ].flatten()  # CONNECTIVITY PENALTY ON THE TRAINING PIXELS
            g_circ = g_circ.reshape(
                (*dataset.raw_img_with_features.shape[:-1], pred.shape[-1])
            )[dataset.labeled_img != unlabeled].flatten()

            # Compute the actual losses and save for later
            l_softmax = loss(
                from_numpy(pred), from_numpy(dtrain.get_label().astype(np.uint8))
            )
            softmax_losses.append(l_softmax)
            # custom_losses.append(l_vf)
            # custom_losses.append(l_conn)
            custom_losses.append(l_circ)

            # g = g_softmax + g_vf
            # g = g_softmax + g_conn
            g = g_softmax + g_circ
            # h = h_softmax + h_vf
            # h = h_softmax + h_conn
            h = h_softmax + h_circ

            model.boost(dtrain, g, h)

        # Evaluate
        yhat_probs = model.predict(dtest).reshape(
            (*dataset.raw_img_with_features.shape[:-1], dataset.n_classes)
        )
        yhat_labels = np.argmax(yhat_probs, axis=-1)
        # Reset to original labels
        yhat_labels = yhat_labels + 1

        softmax_losses_per_lambda[lambd] = softmax_losses
        custom_losses_per_lambda[lambd] = custom_losses
        yhat_probabilities_per_lambda[lambd] = yhat_probs
        yhat_labels_per_lambda[lambd] = yhat_labels

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
    )
