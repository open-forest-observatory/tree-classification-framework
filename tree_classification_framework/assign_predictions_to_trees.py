import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image_level_predictions_file",
        type=Path,
        help="A .json file with image names as keys and classes as values",
    )
    parser.add_argument(
        "output_column_name",
        help="Name of the attribute that the predicted classes will be written to in the output files.",
        type=str,
    )
    parser.add_argument(
        "input_tree_crowns_file", type=Path, help="Path to the detected crowns"
    )
    parser.add_argument(
        "input_tree_tops_file",
        type=Path,
        help="Path to the detected tree tops, which seeded the crowns",
    )
    parser.add_argument(
        "output_tree_crowns_file",
        type=Path,
        help="Path where the tree crowns with the additional classification attributes will be written out",
    )
    parser.add_argument(
        "output_tree_tops_file",
        type=Path,
        help="Path where the tree tops with the additional classification attributes will be written out",
    )

    args = parser.parse_args()
    return args


def compute_per_tree_stats(
    per_view_predictions: pd.Series, output_column_name: str
) -> dict:
    """
    For a given series of predictions, number of views, fair mode, and fraction of predictions
    matching the mode

    Args:
        per_view_predictions (pd.Series): Series of predictions, representing per-view classes
        output_column_name (str): The name of the predicted attribute to use as the key in the output dictionary.

    Returns:
        dict:
            output_column_name: The predicted (modal) class
            "{output_column_name}_frac_matching_mode": The fraction of predictions matching the modal class
            "{output_column_name}_n_preds": The number of predictions

    """
    # Compute the modal class, breaking ties fairly if needed
    modes = per_view_predictions.mode()
    mode = np.random.choice(modes)

    # Count how many view-level predictions contributed
    n_preds = len(per_view_predictions)

    # Determine agreement with final prediction
    frac_matching = (per_view_predictions == mode).sum() / n_preds
    return pd.Series(
        {
            output_column_name: mode,
            f"{output_column_name}_frac_matching_mode": frac_matching,
            f"{output_column_name}_n_preds": n_preds,
        }
    )


def assign_predictions_to_trees(
    output_column_name: str,
    image_level_predictions_file: Path,
    input_tree_crowns_file: Path,
    input_tree_tops_file: Path,
    output_tree_crowns_file: Path,
    output_tree_tops_file: Path,
):
    """Aggregate image-level predictions to per-tree predictions and add them to the crowns and tree tops

    Args:
        output_column_name (str): Name of the attribute column that the predicted classes will be written to
        image_level_predictions_file (Path): A .json file with image names as keys and classes as values
        input_tree_crowns_file (Path): Path to the detected crowns
        input_tree_tops_file (Path): Path to the detected tree tops, which seeded the crowns
        output_tree_crowns_file (Path): Path where the tree crowns with the additional classification attributes will be written out
        output_tree_tops_file (Path): Path where the tree tops with the additional classification attributes will be written out
    """
    # Open the prediction results which contains one class per chip
    with open(image_level_predictions_file, "r") as f:
        preds = json.load(f)

    input_files = list(preds.keys())
    labels = list(preds.values())

    tree_IDs = [Path(f).stem for f in input_files]

    preds_df = pd.DataFrame({"unique_ID": tree_IDs, output_column_name: labels})

    # Determine most commonly predicted class per tree ID and other statistics
    prediction_per_tree = preds_df.groupby(["unique_ID"]).apply(
        lambda x: compute_per_tree_stats(
            x[output_column_name], output_column_name=output_column_name
        ),
        include_groups=False,
    )
    prediction_per_tree = prediction_per_tree.reset_index()

    # Add the predicted attributes to the detected crowns and tree tops
    # Crowns are the easy case because they were used to generate the renders. Therefore, each
    # classified image is named based on the `unique_ID` attribute of the crown it came from.
    # Therefore, the predictions can be easily merged into the crowns by the `unique_ID`
    detected_tree_crowns = gpd.read_file(input_tree_crowns_file)
    detected_tree_crowns["unique_ID"] = detected_tree_crowns["unique_ID"].astype(str)
    detected_tree_crowns = detected_tree_crowns.merge(
        prediction_per_tree, on="unique_ID", how="left"
    )

    ## Tree tops are slightly trickier. Tree tops are used as a seed for deleniating the crown.
    # However, tree tops do not record which crown (if any) is created from them. Instead, this
    # bookkeeping is done by the crown, which records which tree top was used to create it as the
    # treetop_unique_id attribute. Therefore, we use the mapping from unique_ID to treetop_unique_ID
    # to reinterpret the per-crown predictions with respect to the tree tops.
    detected_tree_tops = gpd.read_file(input_tree_tops_file)
    # Ensure the unique_ID column is a string type
    detected_tree_tops["unique_ID"] = detected_tree_tops["unique_ID"].astype(str)

    # Build the mapping from tree crown ID to tree top ID
    crown_to_tree_top_mapping = {
        k: v
        for k, v in zip(
            detected_tree_crowns["unique_ID"].tolist(),
            detected_tree_crowns["treetop_unique_ID"].tolist(),
        )
    }
    # Remap the unique_ID attribute to represent tree top IDs rather than crown IDs. Use map rather
    # than replace so that crown IDs without a corresponding crown become NaN instead of being left
    # as-is, where they could spuriously match an unrelated tree top ID.
    prediction_per_tree.unique_ID = prediction_per_tree.unique_ID.map(
        crown_to_tree_top_mapping
    )
    # Drop predictions that could not be mapped to a tree top
    unmapped = prediction_per_tree.unique_ID.isna()
    if unmapped.any():
        raise ValueError(
            f"{unmapped.sum()} tree predictions did not match any crown and will not be assigned to tree tops"
        )

    # Merge the predictions into the tree tops
    detected_tree_tops = detected_tree_tops.merge(
        prediction_per_tree, on="unique_ID", how="left"
    )

    # create output folders
    output_tree_crowns_file.parent.mkdir(parents=True, exist_ok=True)
    output_tree_tops_file.parent.mkdir(parents=True, exist_ok=True)

    # Save out
    detected_tree_crowns.to_file(output_tree_crowns_file)
    detected_tree_tops.to_file(output_tree_tops_file)


if __name__ == "__main__":
    args = parse_args()

    assign_predictions_to_trees(**args.__dict__)
