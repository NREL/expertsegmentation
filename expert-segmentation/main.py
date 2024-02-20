"""
Main segmentation script.
"""

import argparse
from data import SegDataset, UserInputs
from metrics import *
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
    loss_dict, result_dict, step_dict = run_xgboost(dataset, user_input, save_steps=[25, 40, 55, 75, 95])

    # 3. Display results and metrics
    metrics_df, evaluation_df = print_metrics(result_dict, dataset, user_input)

    if dataset.raw_img.ndim == 2:
        plot_results(result_dict, loss_dict)
    elif dataset.raw_img.ndim == 3:
        save_gifs(result_dict, dataset)

    print()
