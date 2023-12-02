"""
Implementation of XGBoost with custom loss.
"""

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from torch import from_numpy
from torch.nn import CrossEntropyLoss
import xgboost as xgb

from data import SegDataset, UserInputs
from metrics import calculate_volume_fractions, calculate_perc_connected_components


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
    grad[:, 3] = grad[:, 3] * 5
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
    grad = np.stack([grad_row] * len(pred), axis=0)
    return loss, grad.flatten(), 0


def connectivity_obj(pred, lambd, c, H, W, D=None):
    """
    Define "connectivity" loss term to penalize the connectivity of
    a particular class. i.e. the "blue" class (carbon binder) should
    be highly connected.

    Args:
        pred: Predicted probabilities per pixel.
        lambd: Lambda to weight connectivity loss term
        c: The class of interest
        H: Height of original input image
        W: Width of original input image
        D: Depth of original input image, if 3D, else None

    loss =
        lambda * || n_components in class 0 / total n_components || **2

    gradient =
        2 * lambda * || n_components in class 0 / total n_components ||

    """
    if D is None:
        pred_labels = np.argmax(pred, axis=1).reshape((H, W)).astype(np.uint8)
    else:
        pred_labels = np.argmax(pred, axis=1).reshape((H, W, D)).astype(np.uint8)

    perc_connected_components = calculate_perc_connected_components(pred_labels, c=c)
    loss = lambd * perc_connected_components**2
    grad_val = 2 * lambd * perc_connected_components
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
    dtest = xgb.DMatrix(X_test)

    H, W = dataset.raw_img_with_features.shape[:2]

    # Instantiate the model
    model = xgb.Booster(params, [dtrain])

    # Define default loss
    loss = CrossEntropyLoss()

    # Get user inputs
    volume_fraction_targets = user_input.volume_fraction_targets

    # Run model for each lambda
    softmax_losses_per_lambda = dict()
    custom_losses_per_lambda = dict()
    yhat_probabilities_per_lambda = dict()
    yhat_labels_per_lambda = dict()
    for lambd in user_input.lambdas:

        # Run training and save losses (100 epochs)
        softmax_losses = []
        custom_losses = []
        for _ in range(100):
            pred = model.predict(dtrain)
            g_softmax, h_softmax = softmaxobj(
                pred, dtrain
            )  # ONLY FOR THE TRAINING PIXELS (the ones w a label)

            pred_full_image = model.predict(dtrain_full_image)  # ENTIRE IMAGE
            l_vf, g_vf, h_vf = volume_fraction_obj(
                pred_full_image, lambd=lambd, target_distr=volume_fraction_targets
            )  # VOLUME FRACTION PENALTY ON ENTIRE IMAGE
            # l_conn, g_conn, h_conn = connectivity_obj(pred_full_image, lambd, H, W)

            # Re-reduce to only the labeled pixels
            g_vf = g_vf.reshape(
                (*dataset.raw_img_with_features.shape[:-1], pred.shape[-1])
            )[
                dataset.labeled_img != unlabeled
            ].flatten()  # CONNECTIVITY PENALTY ON THE TRAINING PIXELS
            # g_conn = g_conn.reshape((H_train, W_train, pred.shape[-1]))[hand_labels != 42].flatten()  # CONNECTIVITY PENALTY ON THE TRAINING PIXELS

            # Compute the actual losses and save for later
            l_softmax = loss(
                from_numpy(pred), from_numpy(dtrain.get_label().astype(np.uint8))
            )
            softmax_losses.append(l_softmax)
            custom_losses.append(l_vf)

            g = g_softmax + g_vf  # g_conn
            h = h_softmax + h_vf  # h_conn

            model.boost(dtrain, g, h)

        # Evaluate
        yhat_probs = model.predict(dtest)
        yhat_labels = np.argmax(yhat_probs, axis=1)
        # Reset to original labels
        yhat_labels = yhat_labels + 1

        softmax_losses_per_lambda[lambd] = softmax_losses
        custom_losses_per_lambda[lambd] = custom_losses
        yhat_probabilities_per_lambda[lambd] = yhat_probs
        yhat_labels_per_lambda[lambd] = yhat_labels

    return (
        yhat_probabilities_per_lambda,
        yhat_labels_per_lambda,
        softmax_losses_per_lambda,
        custom_losses_per_lambda,
    )
