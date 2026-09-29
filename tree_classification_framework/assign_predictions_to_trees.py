import json
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path

# Parse inputs
column_name = "{{inputs.parameters.column-name}}"
image_level_predictions_file = "{{inputs.parameters.image-level-predictions-file}}"
input_tree_crowns_path = Path("{{inputs.parameters.input-tree-crowns-file}}")
input_tree_tops_path = Path("{{inputs.parameters.input-tree-tops-file}}")
output_tree_crowns_path = Path("{{inputs.parameters.output-tree-crowns-file}}")
output_tree_tops_path = Path("{{inputs.parameters.output-tree-tops-file}}")

# Open the prediction results
with open(image_level_predictions_file, "r") as f:
    preds = json.load(f)

input_files = list(preds.keys())
labels = list(preds.values())

tree_IDs = [Path(f).stem for f in input_files]

preds_df = pd.DataFrame({"tree_ID": tree_IDs, column_name: labels})


def fair_mode(series):
    """Tie break randomly if there two or more options"""
    modes = series.mode()
    mode = np.random.choice(modes)

    n_preds = len(series)

    frac_matching = (series == mode).sum() / n_preds
    return pd.Series(
        {
            column_name: mode,
            f"{column_name}_frac_matching_mode": frac_matching,
            f"{column_name}_n_preds": n_preds,
        }
    )


# Determine most commonly predicted class per tree ID
grouped = preds_df.groupby(["tree_ID"]).apply(
    lambda x: fair_mode(x[column_name]), include_groups=False
)
grouped = grouped.reset_index().rename(columns={"tree_ID": "unique_ID"})

# Add the predicted attributes to the detected crowns and tree tops
## Crowns
detected_tree_crowns = gpd.read_file(input_tree_crowns_path)
detected_tree_crowns["unique_ID"] = detected_tree_crowns["unique_ID"].astype(str)
detected_tree_crowns = detected_tree_crowns.merge(grouped, on="unique_ID", how="left")

## Tree tops
detected_tree_tops = gpd.read_file(input_tree_tops_path)
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
# Remap the unique_ID attribute to represent tree top IDs rather than crown IDs
grouped.unique_ID = grouped.unique_ID.replace(crown_to_tree_top_mapping)
# Merge the predictions into the tree tops
detected_tree_tops = detected_tree_tops.merge(grouped, on="unique_ID", how="left")

# create output folders
output_tree_crowns_path.parent.mkdir(parents=True, exist_ok=True)
output_tree_tops_path.parent.mkdir(parents=True, exist_ok=True)

# Save out
detected_tree_crowns.to_file(output_tree_crowns_path)
detected_tree_tops.to_file(output_tree_tops_path)
