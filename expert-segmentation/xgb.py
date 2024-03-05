"""
Implementation of XGBoost with custom loss.
"""

import GPUtil
import numpy as np
import porespy as ps
from scipy.ndimage import distance_transform_edt
from skimage import measure
from sklearn.preprocessing import OneHotEncoder
from skimage.filters import gaussian
from skimage.segmentation import find_boundaries
from skimage.util import view_as_blocks
from torch import from_numpy
from torch.nn import CrossEntropyLoss
import xgboost as xgb

from data import SegDataset, UserInputs
from metrics import calculate_volume_fractions


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


def circularity_obj_edges(
    pred: np.ndarray,
    lambd: int,
    c: int,
    n_classes: int,
    H: int,
    W: int,
    D: int = None,
):
    """
    loss =
        lambda * || sphericity - 1 ||**2
    gradient =
        2 * lambda * || sphericity - 1 ||
    Args:
        pred: Predicted probabilities per pixel (n_pixels, n_classes)
        lambd: Lambda to weight connectivity loss term
        c: The class of interest
        n_classes: Number of classes
        H: Height of original input image
        W: Width of original input image
        D: Depth of original input image, if 3D, else None
    """
    if D is None:
        pred_labels = np.argmax(pred, axis=1).reshape((H, W)).astype(np.uint8)
    else:
        pred_labels = np.argmax(pred, axis=1).reshape((H, W, D)).astype(np.uint8)
    
    binary = (pred_labels == c).astype(np.uint8)

    # Mark the edges of the target phase
    edges = find_boundaries(binary, mode='inner')

    # If the prediction is all 1 value
    if len(np.unique(binary)) <= 1:
        return np.inf, np.zeros((*pred_labels.shape, n_classes)).flatten(), 0

    window_size = 10

    # Set up arrays to store per-pixel loss and gradient
    gradient_all = np.zeros((*pred_labels.shape, n_classes))
    gradient_all_n = np.ones(gradient_all.shape)
    loss_all = np.zeros(pred_labels.shape)
    loss_all_n = np.ones(loss_all.shape)

    edge_pixels_x, edge_pixels_y, edge_pixels_z = edges.nonzero()
    np.random.seed(42)
    indices_to_include = np.random.choice(list(range(len(edge_pixels_x))),
                                          size=len(edge_pixels_x)//10,
                                          replace=False)
    
    # For each edge pixel
    for i in indices_to_include:
        x = edge_pixels_x[i]
        y = edge_pixels_y[i]
        z = edge_pixels_z[i]

        # Get a subvolume centered on the pixel
        xmin = x - window_size // 2
        xmax = x + window_size // 2
        ymin = y - window_size // 2
        ymax = y + window_size // 2
        zmin = z - window_size // 2
        zmax = z + window_size // 2

        # Check on the edges of the image
        if xmin <= 0:
            xmin = 0
            xmax = window_size - 1
        if ymin <= 0:
            ymin = 0
            ymax = window_size - 1
        if zmin <= 0:
            zmin = 0
            zmax = window_size - 1
        if xmax >= H:
            xmax = H
            xmin = H - window_size + 1
        if ymax >= W:
            ymax = W
            ymin = W - window_size + 1
        if zmax >= D:
            zmax = D
            zmin = D - window_size + 1

        subvolume = binary[xmin : xmax + 1,
                            ymin : ymax + 1,
                            zmin : zmax + 1]

        # Get the connected components within the subvolume
        particles_uniquely_labeled, _ = measure.label(subvolume, return_num=True)

        # Get the sphericity of each subvolume
        props = ps.metrics.regionprops_3D(particles_uniquely_labeled)
        sphericities = [prop.sphericity for prop in props]

        # If the prediction is all 1 value, let the gradient be all zeros
        # and let the loss be nan. These will be ignored in the average loss later
        if len(props) == 0:
            # subvol_loss = np.full(subvolume.shape, np.nan)
            subvol_loss = np.zeros(subvolume.shape)
            subvol_gradient = np.stack([subvol_gradient] * n_classes, axis=-1)

        else:

            # Set up an empty array to store the subvolume gradient
            subvol_gradient = np.zeros(subvolume.shape)
            
            # Let the values of the gradient be the *distance to optimal sphericity (1)*
            # for each uniquely-labeled particle
            labels = [prop.label for prop in props]
            for s, label in enumerate(labels):
                sphericity = sphericities[s]
                subvol_gradient[particles_uniquely_labeled == label] = abs(sphericity - 1)

            # Loss = lambd * || sphericity - 1 ||**2
            subvol_loss = np.mean(lambd * subvol_gradient**2)

            # We only care about the particle *edges*. Set the gradients of any
            # pixels not at the boundaries to 0.
            subvolume_edges = find_boundaries(subvolume)
            subvol_gradient[subvolume_edges == 0] = 0

            # Stack gradients *per-class*
            subvol_gradient = -1 * np.stack([subvol_gradient] * n_classes)
            subvol_gradient[c] *= -1
            subvol_gradient = np.moveaxis(subvol_gradient, 0, -1)
            subvol_gradient = 2 * lambd * subvol_gradient

        # Add the subvolume's gradient to the total gradient,
        # and increase counter to take mean later
        gradient_all[xmin : xmax + 1,
                    ymin : ymax + 1,
                    zmin : zmax + 1] += subvol_gradient
        gradient_all_n[xmin : xmax + 1,
                    ymin : ymax + 1,
                    zmin : zmax + 1] += 1
        
        # Same for loss
        loss_all[xmin : xmax + 1,
                ymin : ymax + 1,
                zmin : zmax + 1] += subvol_loss
        loss_all_n[xmin : xmax + 1,
                ymin : ymax + 1,
                zmin : zmax + 1] += 1
    

    # Take average of all the overlapping gradients / losses
    gradient_all = gradient_all / gradient_all_n
    loss_all = loss_all / loss_all_n        

    # Smooth the gradient to spread the effect at the edges
    gradient_all = gaussian(gradient_all, channel_axis=-1)

    # Return average pixelwise loss, ignoring nans
    return np.nanmean(loss_all), gradient_all.flatten(), 0


def circularity_obj(
    pred: np.ndarray,
    lambd: int,
    c: int,
    n_classes: int,
    H: int,
    W: int,
    D: int = None,
):
    """
    loss =
        lambda * || sphericity - 1 ||**2
    gradient =
        2 * lambda * || sphericity - 1 ||
    Args:
        pred: Predicted probabilities per pixel (n_pixels, n_classes)
        lambd: Lambda to weight connectivity loss term
        c: The class of interest
        n_classes: Number of classes
        H: Height of original input image
        W: Width of original input image
        D: Depth of original input image, if 3D, else None
    """
    if D is None:
        pred_labels = np.argmax(pred, axis=1).reshape((H, W)).astype(np.uint8)
    else:
        pred_labels = np.argmax(pred, axis=1).reshape((H, W, D)).astype(np.uint8)
    
    binary = (pred_labels == c).astype(np.uint8)

    window_size = 15  # TODO make generalizable

    # 15, 15, 15
    # 5, 5, 5
    # 25, 25, 25

    # TODO MAKE THE BLOCK SIZE GENERALIZABLE !!
    # subvolumes = view_as_blocks(binary, block_shape=(30, 29, 29))
    subvolumes = view_as_blocks(binary, block_shape=(25, 25, 25))
    s1, s2, s3, _, _, _ = subvolumes.shape
    # subvolumes = subvolumes.reshape(s1*s2*s3, 30, 29, 29)
    subvolumes = subvolumes.reshape(s1*s2*s3, 25, 25, 25)
    gradient_all = np.zeros((*subvolumes.shape, 2))
    loss_all = np.zeros(subvolumes.shape)

    for j, subvolume in enumerate(subvolumes):

        particles_uniquely_labeled, _ = measure.label(subvolume, return_num=True)

        props = ps.metrics.regionprops_3D(particles_uniquely_labeled)
        sphericities = [prop.sphericity for prop in props]

        # Set up an empty array to store the gradient
        gradient = np.zeros(subvolume.shape)

        # If the prediction is all 1 value, let the gradient be all zeros
        # and let the loss be nan. These will be ignored in the average loss later
        if len(props) == 0:
            loss = np.full(subvolume.shape, np.nan)
            gradient = np.stack([gradient] * n_classes, axis=-1)

            loss_all[j] = loss
            gradient_all[j] = gradient
            continue
        
        # Let the values of the gradient be the *distance to optimal sphericity (1)*
        # for each uniquely-labeled particle
        labels = [prop.label for prop in props]
        for i, label in enumerate(labels):
            sphericity = sphericities[i]
            gradient[particles_uniquely_labeled == label] = abs(sphericity - 1)

        # Loss = lambd * || sphericity - 1 ||**2
        loss = np.mean(lambd * gradient**2)

        # We only care about the particle *edges*. Set the gradients of any
        # pixels not at the boundaries to 0.
        edges = find_boundaries(subvolume)
        gradient[edges == 0] = 0

        # There needs to be a gradient *per-class*. To erode, set the gradient of the 
        # *target phase* to be positive, and the gradient of the other classes negative.
        gradient = -1 * np.stack([gradient] * n_classes)
        gradient[c] *= -1
        gradient = np.moveaxis(gradient, 0, -1)
        gradient = 2 * lambd * gradient

        gradient_all[j] = gradient
        loss_all[j] = loss

    loss_all = loss_all.reshape(pred_labels.shape)
    gradient_all = gradient_all.reshape((*pred_labels.shape, n_classes))

    gradient_all = gaussian(gradient_all, channel_axis=-1)

    return np.nanmean(loss_all), gradient_all.flatten(), 0


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

    num_epochs = user_input.num_epochs

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
    model_default_loss = xgb.train(params, dtrain, num_epochs)
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

        # Run training and save losses
        softmax_losses = []
        custom_losses = []
        step_predictions = dict()
        for i in range(num_epochs):
            if i % 10 == 0:
                print(f"\tepoch {i} / {num_epochs - 1}")
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
