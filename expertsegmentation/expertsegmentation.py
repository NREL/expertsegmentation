import GPUtil
import numpy as np
from torch import from_numpy
from torch.nn import CrossEntropyLoss
from typing import Union
import xgboost as xgb
import warnings

from expertsegmentation.evaluate import *
from expertsegmentation.objectives import *
from expertsegmentation.named_constants import *


# TODO at segmentation time check that dataset's labels match number of labels provided in target
# TODO for conn check that the target class provided isn't out of range


class SegmentModel:
    """
    Base model class for segmentation.

    Args:
        objective (str, list[str], set[str]): Name of target property
        target (dict, int): Value of target property. If objective is 
                'volume_fraction', should be a dictionary of target fraction
                per class. If objective is 'connectivity', should be the
                target class label.
        direction (str): Used if objective is 'connectivity'. One of {'min', 'max'}.
        lambd (float, list[float], dict): list of lambda values to weight the target property.
        n_epochs (int): Number of training epochs to run
        save_steps (list[int]): List of epoch numbers at which intermediate output is saved.

    """
    def __init__(self,
                 objective: Union[str, list[str], set[str]],
                 target: Union[dict, int],
                 direction: str = None,
                 lambd: Union[float, list[list[float]]] = None,
                 n_epochs: int = 100,
                 save_steps: list[int] = [5, 25, 50, 75, 95]) -> None:
        
        # Attributes, populated below
        self.objectives = None
        self.target_vf = None
        self.target_conn = None
        self.direction = None
        self.lambdas = None
        self.n_lambdas = None

        self.n_epochs = n_epochs
        self.save_steps = save_steps

        if objective == "volume_fraction":
            self.objectives = [objective]
            # Dictionary keyword is unnecessary
            if direction is not None:
                warnings.warn("Unused parameter `direction`.")
            self.target_vf = self._check_vf_inputs(target)            
            if lambd is None:
                self.lambdas = [DEFAULT_LAMBDA]
            elif isinstance(lambd, float) or isinstance(lambd, int):
                self.lambdas = {objective: [lambd]}
            else:  # It's a list
                self.lambdas = {objective: lambd}
            self.n_lambdas = len(self.lambdas[objective])

        elif objective == "connectivity":
            self.objectives = [objective]
            self.target_conn, self.direction = self._check_conn_inputs(target, direction)
            if lambd is None:
                self.lambdas = [DEFAULT_LAMBDA]
            elif isinstance(lambd, float):
                self.lambdas = {objective: [lambd]}
            else:
                self.lambdas = {objective: lambd}
            self.n_lambdas = len(self.lambdas[objective])

        elif isinstance(objective, list) or isinstance(objective, set):
            self.objectives = objective
            if not isinstance(target, dict):
                raise ValueError("Received multiple objectives. Expected target to be a <dict>.")
            if lambd is None:
                lambd = {key: [DEFAULT_LAMBDA] for key in objective}
            if not isinstance(lambd, dict):
                raise ValueError("Received multiple objectives. Expected lambd to be a <dict>.")
            for obj in objective:
                if obj not in lambd:
                    raise ValueError(f"{obj} not found in `lambd` dictionary.")
                if isinstance(lambd[obj], float):
                    lambd[obj] = [lambd[obj]]
                if not isinstance(lambd[obj], list) and not isinstance(lambd[obj], set):
                    raise ValueError(f"Expected lambd['{obj}'] to be a <list> or <set>.")
                list_lengths = [len(lambd[key]) for key in lambd]
                if len(set(list_lengths)) != 1:
                    raise ValueError(f"Non-matching list lengths of lambda values.")
                self.n_lambdas = list_lengths[0]
                self.lambdas = lambd
                if obj == "volume_fraction":
                    self.target_vf = self._check_vf_inputs(target[obj])
                elif obj == "connectivity":
                    self.target_conn, self.direction = self._check_conn_inputs(target[obj], direction)
                else:
                    raise ValueError(f"Unsupported objective. Supported objectives include {SUPPORTED_OBJECTIVES}. Received {obj}.")

        else:
            raise ValueError(f"Unsupported objective. Supported objectives include {SUPPORTED_OBJECTIVES}. Received {objective}.")

        
    def _check_vf_inputs(self, target):
        # Target should be a dictionary
        if not isinstance(target, dict):
            raise ValueError("Expected target to be a dictionary for objective 'volume_fraction'.")

        # Get target distribution from dictionary,
        # which should have a target fraction per class that sums to 1.
        n_classes = len(target)
        target_array = []
        for i in range(1, n_classes+1):
            try:
                target_array.append(target[i])
            except:
                raise ValueError("Incorrect target provided. Ensure that labels are {1, 2, ... n_classes}.")

        target_array = np.array(target_array)
        self._check_vf(target_array)
        return target_array


    def _check_vf(self, target_array):
        # Make sure that volume fraction targets sum to 1
        vf_sum = target_array.sum()
        if vf_sum != 1:
            raise ValueError(
                f"Volume fraction targets must sum to 1. Received targets that sum to {vf_sum}."
            )
     
    def _check_conn_inputs(self, target, direction):
        if not isinstance(target, int):
            raise ValueError(f"Expected target to be type `int` for target 'connectivity'.")
        if direction is None:
            raise UserWarning(f"Missing keyword `direction` for target 'connectivity'.")
        if direction not in {'min', 'max'}:
            raise ValueError(f"Expected direction to be one of {'min', 'max'}. Received %s" %(direction))        
        return target, direction


    def segment(self, dataset: SegDataset) -> None:
        """Main function to fit XGBoost model and make predictions.

        Args:
            dataset (SegDataset): Object with raw input and labeled image.

        Returns:
            result_dict (dict): Dictionary with results using default loss, custom loss (if applicable),
                                and intermediate steps (if applicable)
            loss_dict (dict):   Dictionary with softmax and custom losses during training with custom loss.
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

        self.n_classes = dataset.n_classes

        # The unlabeled pixels are 0. For XGBoost, the class indices
        # should start from 0, so redefine "unlabeled" to be the largest index.
        dataset.labeled_img[dataset.labeled_img == 0] = dataset.n_classes + 1
        dataset.labeled_img = dataset.labeled_img - 1
        unlabeled = dataset.n_classes

        # Select the labeled pixels for training
        X_train = dataset.img_with_features[dataset.labeled_img != unlabeled]
        y_train = dataset.labeled_img[dataset.labeled_img != unlabeled]

        # Reshape the image into feature matrices with shape (H*W, C) or (H*W*D, C)
        X_test = dataset.img_with_features.reshape(
            -1, dataset.img_with_features.shape[-1]
        )

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtrain_full_image = xgb.DMatrix(X_test)
        dtest = dtrain_full_image  # xgb.DMatrix(X_test)

        # Run default loss first
        print("Evaluating with native loss...")
        model_default_loss = xgb.train(params, dtrain, self.n_epochs)
        yhat_probs_default = model_default_loss.predict(dtest)
        yhat_probs_default = yhat_probs_default.reshape(
            (*dataset.img_with_features.shape[:-1], dataset.n_classes)
        )
        # Add 1 to map back to original labels
        yhat_labels_default = np.argmax(yhat_probs_default, axis=-1) + 1

        result_dict = {
            "probabilities_default_loss": yhat_probs_default,
            "labels_default_loss": yhat_labels_default,
        }

        self.loss_dict = dict()
        self.step_dict = dict()
        self.result_dict = result_dict

        if self.objectives is None:
            return self.result_dict, self.loss_dict
        
        # Instantiate the model
        model = xgb.Booster(params, [dtrain])

        # Define default loss
        loss = CrossEntropyLoss()

        # Run model for each lambda
        softmax_losses_per_lambda = dict()
        custom_losses_per_lambda = dict()
        yhat_probabilities_per_lambda = dict()
        yhat_labels_per_lambda = dict()
        step_dict = dict()

        for l in range(self.n_lambdas):

            if len(self.objectives) == 1:
                print("Evaluating for lambda = {}".format(self.lambdas))
            else:
                print("Evaluating for lambda = {}".format({key: self.lambdas[key][l] for key in self.objectives}))

            # Run training and save losses
            softmax_losses = []
            custom_losses = []
            step_predictions = dict()
            for i in range(self.n_epochs):
                if i % 10 == 0:
                    print(f"\tepoch {i} / {self.n_epochs - 1}")
                pred = model.predict(dtrain)

                # Softmax loss ONLY FOR THE TRAINING PIXELS (the ones w a label)
                g_softmax, h_softmax = softmaxobj(pred, dtrain)

                # Custom loss across the entire image
                pred_full_image = model.predict(dtrain_full_image)

                l_custom_sum = 0
                g_custom_sum = []
                h_custom_sum = 0
                for objective in self.objectives:
                    if objective == "volume_fraction":
                        l_custom, g_custom, h_custom = volume_fraction_obj(
                            pred_full_image,
                            self.lambdas["volume_fraction"][l],
                            target_distr=self.target_vf,
                        )
                    elif objective == "connectivity":
                        l_custom, g_custom, h_custom = connectivity_obj(
                            pred_full_image,
                            self.lambdas["connectivity"][l],
                            self.target_conn - 1,  # TODO
                            dataset.n_classes,
                            *dataset.img.shape,
                            self.direction,  # TODO
                        )
                    
                    l_custom_sum += l_custom
                    g_custom_sum.append(g_custom)
                    h_custom_sum += h_custom

                # Aggregate
                g_custom_sum = np.sum(g_custom_sum, axis=0)

                # Re-reduce to only the labeled pixels
                g_custom_before_reshape = g_custom
                g_custom = g_custom.reshape(
                    (*dataset.img_with_features.shape[:-1], pred.shape[-1])
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
                if i in self.save_steps:
                    step_predictions[i] = {'prediction': pred_full_image.reshape((*dataset.img_with_features.shape[:-1], dataset.n_classes)),
                                        'l_softmax': l_softmax,
                                        'l_custom': l_custom,
                                        'g_softmax': g_softmax,
                                        'g_custom': g_custom_before_reshape,
                    }

            # Evaluate
            yhat_probs = model.predict(dtest).reshape(
                (*dataset.img_with_features.shape[:-1], dataset.n_classes)
            )
            yhat_labels = np.argmax(yhat_probs, axis=-1)
            # Reset to original labels
            yhat_labels = yhat_labels + 1

            lambd_str = "{}".format({key: self.lambdas[key][l] for key in self.objectives})
            softmax_losses_per_lambda[lambd_str] = softmax_losses
            custom_losses_per_lambda[lambd_str] = custom_losses
            yhat_probabilities_per_lambda[lambd_str] = yhat_probs
            yhat_labels_per_lambda[lambd_str] = yhat_labels
            step_dict[lambd_str] = step_predictions

        self.loss_dict["custom_losses"] = custom_losses_per_lambda
        self.loss_dict["softmax_losses"] = softmax_losses_per_lambda
        self.result_dict["probabilities_custom_loss"] = yhat_probabilities_per_lambda
        self.result_dict["labels_custom_loss"] = yhat_labels_per_lambda
        self.result_dict["intermediate_epoch_results"] = step_dict
        self.step_dict = step_dict

        return self.result_dict, self.loss_dict


    def plot_results(self, dataset: SegDataset, slice_idx=None, save_fn: str = None):
        """
        Visualize results. For each lambda, plot the raw image, labels, difference
        to segmentation with naive loss, and loss curves.

        Args:
            dataset (SegDataset):  Dataset used for training.
            slice_idx (int):       If dataset is a 3D volume, used to plot a specific slice.
                                   If not provided, plot the first slice.
            save_fn (str):         Filepath to save resulting image, if provided.


        """
        # If a 3D volume is provided but no slice index, plot slice 0
        if len(dataset.img.shape) > 2 and slice_idx == None:
            slice_idx = 0
        if self.result_dict is None:
            raise Exception("No results to plot. Try running 'SegmentModel.segment()' first.")
        fig = plot(result_dict=self.result_dict,
                     loss_dict=self.loss_dict,
                     img=dataset.img,
                     n_epochs=self.n_epochs,
                     slice_idx_3d=slice_idx)
        if save_fn is not None:
            fig.savefig(save_fn)
        fig.show()


    def print_metrics(self):
        """
        Print table of physical metrics on results.
        """
        if self.result_dict is None:
            raise Exception("No results to plot. Try running 'SegmentModel.segment()' first.")
        _ = print_metrics_table(self.result_dict, self.n_classes, self.objectives, self.target_vf)


    def plot_steps(self, dataset, slice_idx=0, lambd: Union[int, float] = None, save_fn: str = None):
        """
        Visualize intermediate results during training. For the given lambda (or the first lambda
        if None provided) that was used during training, plot the raw image, labels, difference
        to segmentation with naive loss, and loss curves.

        Args:
            dataset (SegDataset):  Dataset used for training.
            slice_idx (int):       If dataset is a 3D volume, used to plot a specific slice.
                                   If not provided, plot the first slice.
            lambd (int, float):    Lambda of interest that was used during training to plot results for.
            save_fn (str):         Filepath to save resulting image, if provided.
        """
        fig = plot_steps_3d_slice(self.result_dict,
                            img=dataset.img,
                            slice_idx=slice_idx,
                            lambd=lambd)
        if save_fn is not None:
            fig.savefig(save_fn)
        fig.show()
