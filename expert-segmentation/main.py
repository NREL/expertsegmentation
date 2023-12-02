"""
Main segmentation script.
"""

import argparse
from data import SegDataset, UserInputs
from metrics import plot_results, print_metrics
from xgb import run_xgboost


parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input_image",
    type=str,
    required=True,
    help="Path to the input image for segmentation",
)
parser.add_argument(
    "-l",
    "--labeled_image",
    type=str,
    required=True,
    help="Path to the input image with hand-labels to train on",
)
parser.add_argument(
    "-f",
    "--input_image_with_features",
    type=str,
    required=False,
    help="Path to the input image with transformation features already computed",
)
parser.add_argument(
    "-u",
    "--user_input",
    type=str,
    required=False,
    default="expert-segmentation/data/user_input.json",
    help="Path to json file that defines expert-informed targets",
)


if __name__ == "__main__":

    # 1. Load data and compute input features
    args = parser.parse_args()
    user_input = UserInputs(args.user_input)
    dataset = SegDataset(
        raw_img_fn=args.input_image,
        labeled_img_fn=args.labeled_image,
        filter_dict=user_input.filters,
        raw_img_with_features_fn=args.input_image_with_features,
    )

    # 2. Run segmentation with targets
    yhat_probabilities, yhat_labels, softmax_losses, custom_losses = run_xgboost(
        dataset, user_input
    )

    # 3. Display results and metrics
    plot_results(yhat_labels)
    print_metrics(yhat_labels, dataset, user_input)
    print()

    """
    # TODO
    - circularity/sphericity
    - run multiple targets at a time? right now just one at a time
    """
