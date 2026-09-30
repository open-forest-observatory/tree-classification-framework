import argparse
from pathlib import Path
import json

import geopandas as gpd
import numpy as np

# Consume a folder of chipped images corresponding to one dataset.
# Produce a json folder assigning each one a classification
# The per-view predictions should be structured, corresponding to a per-tree label.
# This label can either be randomly generated or derived from a reference file.
FRACTION_MATCHING_MODE = 0.75
UNIQUE_ID_COLUMN = "unique_ID"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "chips_folder", type=Path, help="Path to folder of per-tree chips"
    )
    parser.add_argument(
        "output_file",
        type=Path,
        help="Path to write simulated predictions as a .json file",
    )
    parser.add_argument(
        "--reference-file",
        type=Path,
        help="A geopandas-loadable file with `unique_ID` column that is used to assign labels to trees.",
    )
    parser.add_argument(
        "--reference-attribute",
        type=str,
        help="If --reference-file is provided, this attribute of the file is used",
    )
    parser.add_argument(
        "--class-list",
        nargs="+",
        type=str,
        help="Classes to use in the case of random classification",
    )
    parser.add_argument(
        "--fraction-matching-mode",
        type=float,
        default=FRACTION_MATCHING_MODE,
        help="In the case of random classification, this is the likelihood that each view-level prediction will match the randomly-assigned tree-level class",
    )
    args = parser.parse_args()

    return args


def simulate_predictions(
    chips_folder: Path,
    output_file: Path,
    reference_file: Path | None = None,
    reference_attribute: str | None = None,
    class_list: list[str] | None = None,
    fraction_matching_mode: float = FRACTION_MATCHING_MODE,
):
    """Simulate view-level predictions for a folder of per-tree chips and write them to a .json file

    Args:
        image_folder (Path): Path to folder of per-tree chips
        output_file (Path): Path to write simulated predictions as a .json file
        reference_file (Path | None, optional): A geopandas-loadable file with `unique_ID` column that is used to assign labels to trees. Defaults to None.
        reference_attribute (str | None, optional): If reference_file is provided, this attribute of the file is used. Defaults to None.
        class_list (list[str] | None, optional): Classes to use in the case of random classification. Defaults to None.
        fraction_matching_mode (float | None, optional): In the case of random classification, this is the likelihood that each view-level prediction will match the randomly-assigned tree-level class. Defaults to None.
    """
    # Basic argument checks
    if reference_file is not None and reference_attribute is None:
        raise ValueError(
            f"Cannot provide reference file without a `reference_attribute`"
        )

    if reference_file is None and class_list is None:
        raise ValueError(
            "If no reference file is provided the class list must be provided instead"
        )

    chip_files = [f for f in chips_folder.rglob("*") if f.is_file()]

    chip_IDs = [f.stem for f in chip_files]

    unique_IDs = np.unique(chip_IDs).tolist()

    # Parse or simulate the tree per ID
    if reference_file is not None:
        reference_gpd = gpd.read_file(reference_file)

        IDs_to_class = {
            k: v
            for k, v in zip(
                reference_gpd[UNIQUE_ID_COLUMN], reference_gpd[reference_attribute]
            )
        }
        # TODO Check that ll unique IDs are present as keys

        class_list = np.unique(list(IDs_to_class.values())).tolist()
    else:
        IDs_to_class = {ID: np.random.choice(class_list) for ID in unique_IDs}

    output_classes = {
        chip_file: IDs_to_class[chip_ID]
        for chip_file, chip_ID in zip(chip_files, chip_IDs)
    }

    # Correct for the fact that we might re-pick the existing ID, so sample n_classes / (n_classes - 1) times more frequently
    adjusted_threshold = np.clip(
        1 - ((1 - fraction_matching_mode) * len(class_list) / (len(class_list) - 1)),
        0,
        1,
    )
    output_classes = {
        str(k): str(
            v if np.random.rand() < adjusted_threshold else np.random.choice(class_list)
        )
        for k, v in output_classes.items()
    }

    output_file.parent.mkdir(exist_ok=True, parents=True)
    with open(output_file, "w") as output_file_h:
        json.dump(output_classes, output_file_h)


if __name__ == "__main__":
    args = parse_args()

    simulate_predictions(**args.__dict__)
