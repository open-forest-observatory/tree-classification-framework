import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
import argparse
import warnings

import geopandas as gpd
import pandas as pd

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
ID_COLUMN = "unique_ID"
LIVE_DEAD_ATTRIBUTE = "live_dead_prediction"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reorganize chips into a train/val folder structure for classification. Chips begin structured according to the original folder of images used in photogrammetry and are remapped to a flat folder for each class within the train/val split."
    )
    parser.add_argument(
        "chips_dir",
        type=Path,
        help="Input chips folder. This should be organized by dataset_ID at the top level, and then each folder within it should be organized in the same structure as the original photogrammetry imagery",
    )
    parser.add_argument(
        "metadata_dir",
        type=Path,
        help="Input tree-level metadata folder. There should be one file per dataset and each file should be named based on the dataset_ID with the `.gpkg` extension",
    )
    parser.add_argument(
        "train_val_split_file",
        type=Path,
        help="Path to a .csv defining the train/val split. The file should contain two columns without headers. The first column should be the dataset_ID and the second is 'train' or 'val'",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory to write reorganized chips to. The top level folders will be 'train' or 'val' and then each will contain subfolders for each class.",
    )
    parser.add_argument(
        "--class-remap-file",
        type=Path,
        default=None,
        help="JSON file mapping original attribute values to final class names",
    )
    parser.add_argument(
        "--attribute-to-train-on",
        type=str,
        default="species_code",
        help="Column name in the geopackage to use as the class label",
    )
    parser.add_argument(
        "--filter-dead-trees",
        action="store_true",
        help="If set, drop trees predicted as dead based on the --dead-trees-attribute column",
    )
    parser.add_argument(
        "--live-dead-attribute",
        type=str,
        default=LIVE_DEAD_ATTRIBUTE,
        help="Column name in the geopackage indicating whether a tree is live or dead (default: live_dead_predicted)",
    )
    return parser.parse_args()


def link_or_copy(src, dst):
    """Hardlink src -> dst, falling back to a copy if they are on different filesystems or this is not supported."""
    dst.parent.mkdir(exist_ok=True, parents=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def reorganize_single_dataset(
    chips_folder: Path,
    tree_metadata_path: Path,
    dataset_name: str,
    output_dir: Path,
    training_attribute: str,
    class_counters: dict,
    unmapped_classes: dict,
    reorganization_summary: list,
    class_name_remapping: dict | None,
    live_dead_attribute: str,
    filter_dead_trees: bool,
):
    """
    Restructure the chips from a single dataset to be placed in per-class folders and have
    sequential IDs

    Args:
        chips_folder (Path): Path to one dataset worth of chips
        tree_metadata_path (Path): Path to the per-tree metadata
        dataset_name (str): The name of the dataset, often the drone-plot pair
        output_dir (Path): Where to write the class-level reorganized chips
        training_attribute (str): What attribute from the metadata to reorganize based on
        class_counters (dict): Counters for number of remapped chips per class
        unmapped_classes (dict): Counters for number of chips skipped because they weren't in the list of included classes
        reorganization_summary (list): A running list tracking the reorganization to use downstream for tree-level metrics
        class_name_remapping (dict | None): A dictionary mapping from original to new class names. If None, all classes will be kept.
        live_dead_attribute (str): What attribute determines if a tree is live or dead
        filter_dead_trees (bool): Should chips from dead trees be skipped in the linking process

    Raises:
        ValueError:
            If the input chip folder does not exist
        ValueError:
            If the required training attribute column, live/dead attribute, or the unique ID for a
            chip is not included in the metadata file
    """
    if not chips_folder.is_dir():
        raise ValueError(f"[{chips_folder}] does not exist")

    # Load the geopackage and remap the class attribute for each tree
    tree_metadata = gpd.read_file(tree_metadata_path)

    # Check columns
    for required in (
        (ID_COLUMN, training_attribute) + (live_dead_attribute,)
        if filter_dead_trees
        else ()
    ):
        if required not in tree_metadata.columns:
            raise ValueError(
                f"[{dataset_name}] column '{required}' not found in {tree_metadata_path}; "
                f"available columns: {list(tree_metadata.columns)}"
            )
    tree_metadata[ID_COLUMN] = tree_metadata[ID_COLUMN].astype(str)
    # Drop any rows with critical elements that are na
    tree_metadata.dropna(axis=0, subset=[ID_COLUMN, training_attribute])

    # Drop rows which are not predicted as live if requested
    if filter_dead_trees:
        # Note that anything other than the literal 'Live' is dropped
        live_trees = tree_metadata[live_dead_attribute] == "Live"
        dead_tree_IDs = set(tree_metadata.loc[~live_trees, ID_COLUMN].tolist())
        tree_metadata = tree_metadata[live_trees]
        print(
            f"[{dataset_name}] Dropped {(~live_trees).sum()} of {len(live_trees)} trees predicted as dead from {dataset_name}"
        )
    else:
        dead_tree_IDs = set()

    # Create the mapping from an individual tree's ID to the attribute listed in the metadata file
    id_to_attribute = dict(
        zip(tree_metadata[ID_COLUMN], tree_metadata[training_attribute])
    )

    if class_name_remapping is None:
        # Keep the class the same as the attribute
        id_to_class = id_to_attribute
    else:
        # Create a mapping which composes first mapping from id to attribute then attribute to class
        # IDs who's attributes do not correspond to a key in class_name_remapping are not included
        # in the composed mapping
        id_to_class = {
            id: class_name_remapping[attribute]
            for id, attribute in id_to_attribute.items()
            if attribute in class_name_remapping
        }

    # List all files
    chip_paths = [
        chip_path
        for chip_path in chips_folder.rglob("*")
        if (chip_path.is_file() and chip_path.suffix.lower() in IMAGE_EXTS)
    ]

    # Guarantee that every chip has an entry in the metadata file
    unique_ids = set([chip_path.stem for chip_path in chip_paths])
    if missing_ids := (
        unique_ids - set(tree_metadata[ID_COLUMN].unique()).union(dead_tree_IDs)
    ):
        raise ValueError(
            f"The following unique IDs were obtained from chips but had no corresponding row in the tree metadata {missing_ids}"
        )

    # Count chips which no longer have a class due to the remapping
    if class_name_remapping is not None and (
        missing_ids := (unique_ids - set(list(id_to_class.keys())).union(dead_tree_IDs))
    ):
        for missing_id in missing_ids:
            missing_attribute = id_to_attribute[missing_id]
            unmapped_classes[missing_attribute] += 1

    # Hardlink/copy every chip to the output folder for its remapped class
    for chip_path in chip_paths:
        unique_id = chip_path.stem
        output_class = id_to_class.get(unique_id)

        # This unique ID did not have a class, because the corresponding class was dropped during
        # the remapping process.
        if output_class is None:
            continue

        reorganized_chip_path = (
            output_dir
            / output_class
            / f"{class_counters[output_class]:06d}{chip_path.suffix.lower()}"
        )

        link_or_copy(chip_path, reorganized_chip_path)
        # Increment class counter
        class_counters[output_class] += 1

        # Add an entry to the list which tracks the reorganization
        reorganization_summary.append(
            {
                "image_path": str(reorganized_chip_path),
                "class": output_class,
                "tree_id": unique_id,
                "dataset_id": dataset_name,
            }
        )
    print(
        f"[{dataset_name}] linked {sum(class_counters.values())} chips "
        f"from {len(id_to_class)} matched trees"
    )


def main(
    chips_dir: Path,
    metadata_dir: Path,
    train_val_split_file: Path,
    output_dir: Path,
    class_remap_file: Path | None = None,
    attribute_to_train_on: str = "species_code",
    filter_dead_trees: bool = False,
    live_dead_attribute: str = "live_dead_prediction",
):
    """
    Reorganize chips into a train/val folder structure for classification.

    Args:
        chips_dir (Path):
            Input chips folder. This should be organized by dataset_ID at the top level, and then
            each folder within it should be organized in the same structure as the original
            photogrammetry imagery.
        metadata_dir (Path):
            Input tree-level metadata folder. There should be one file per dataset and each file
            should be named based on the dataset_ID with the `.gpkg` extension.
        train_val_split_file (Path):
            Path to a .csv defining the train/val split. The file should contain two columns
            without headers. The first column should be the dataset_ID and the second is 'train'
            or 'val'.
        output_dir (Path):
            Output directory to write reorganized chips to. The top level folders will be 'train'
            or 'val' and then each will contain subfolders for each class. Any existing contents
            are deleted.
        class_remap_file (Path | None, optional):
            JSON file mapping original attribute values to final class names. If None, every
            class is kept unchanged. Defaults to None.
        attribute_to_train_on (str, optional):
            Column name in the geopackage to use as the class label. Defaults to "species_code".
        filter_dead_trees (bool, optional):
            If set, drop trees predicted as dead based on the `dead_trees_attribute` column.
            Defaults to False.
        live_dead_attribute (str, optional):
            Column name in the geopackage indicating whether a tree is live or dead. Defaults to
            "predicted_health_status".
    """
    # Load the class remapping (original attribute value -> final class name). With no file
    # provided, keep all classes unchanged
    if class_remap_file:
        with open(class_remap_file, "r") as f:
            class_remapping = {str(k): str(v) for k, v in json.load(f).items()}
        print(
            f"Loaded class remapping: {len(class_remapping)} attribute values -> "
            f"{len(set(class_remapping.values()))} output classes"
        )
    else:
        class_remapping = None
        print("No class remapping file provided; keeping every class unchanged")

    # Load the definition of which dataset to assign to train and val
    train_val_split = pd.read_csv(
        train_val_split_file,
        names=("dataset_id", "train_val"),
        skipinitialspace=True,
        dtype=str,
    )

    if not train_val_split.train_val.isin(["train", "val"]).all():
        raise ValueError(
            f"The train_val column should only have values of 'train' or 'val' but instead has {train_val_split.train_val.unique().tolist()}"
        )

    # Error if the same dataset appears multiple times within the same fold
    if (
        len(
            duplicated_rows := train_val_split[train_val_split.duplicated(keep="first")]
        )
        > 0
    ):
        raise ValueError(
            "The following dataset_IDs are listed more than once within the same fold: "
            f"{list(duplicated_rows.itertuples(index=False, name=None))}"
        )

    train_datasets = train_val_split.query("train_val=='train'").dataset_id.to_list()
    val_datasets = train_val_split.query("train_val=='val'").dataset_id.to_list()

    if len(overlapping_IDs := set(train_datasets).intersection(set(val_datasets))) > 0:
        # Note that this is a warning rather than an error because these datasets may be
        # intentionally replicated, such as when testing the code or evaluating in-distribution
        # performance.
        warnings.warn(
            f"The following dataset_IDs overlap between train and val: {list(overlapping_IDs)}"
        )

    # Delete the output directory if previously created
    if output_dir.is_dir():
        shutil.rmtree(output_dir)

    # Reorganize train and val datasets independently
    for train_val, dataset_IDs in (("train", train_datasets), ("val", val_datasets)):

        # This saves information about the restructuring process which is used later for tree-level
        # metrics
        reorganization_summary = []
        # Output dir
        dataset_output_dir = output_dir / train_val

        # Count the number of chips written per class and the number per class that were skipped
        class_counters = defaultdict(int)
        unmapped_classes = defaultdict(int)

        # Iterate over each dataset
        for dataset_ID in dataset_IDs:
            dataset_chips_dir = chips_dir / dataset_ID
            tree_metadata_file = metadata_dir / f"{dataset_ID}.gpkg"

            # Call per-folder reorganization
            reorganize_single_dataset(
                chips_folder=dataset_chips_dir,
                tree_metadata_path=tree_metadata_file,
                dataset_name=dataset_ID,
                output_dir=dataset_output_dir,
                training_attribute=attribute_to_train_on,
                class_counters=class_counters,
                unmapped_classes=unmapped_classes,
                reorganization_summary=reorganization_summary,
                class_name_remapping=class_remapping,
                live_dead_attribute=live_dead_attribute,
                filter_dead_trees=filter_dead_trees,
            )

        # Print summary
        total = sum(class_counters.values())
        print(f"\n=== Reorganization summary ({train_val})===")
        for output_class in sorted(class_counters):
            print(f"  {output_class}: {class_counters[output_class]} chips")
        print(f"  TOTAL: {total} chips across {len(class_counters)} class(es)")
        if unmapped_classes:
            print(
                "Chips skipped (attribute value null or absent from the remapping file):"
            )
            for raw, count in sorted(unmapped_classes.items(), key=lambda kv: -kv[1]):
                print(f"  {raw!r}: {count}")

        if total == 0:
            raise ValueError(
                "No chips were written; check attribute_to_train_on and the remapping file"
            )

        # Save out the summary file
        reorganization_summary = pd.DataFrame(reorganization_summary)
        reorganization_summary_path = Path(
            output_dir, f"{train_val}_reorganization_summary.csv"
        )
        reorganization_summary.to_csv(reorganization_summary_path, index=False)

    # Ensure matching train and val folders
    train_class_names = {p.name for p in (output_dir / "train").glob("*")}
    val_class_names = {p.name for p in (output_dir / "val").glob("*")}

    val_not_in_train = val_class_names - train_class_names
    train_not_in_val = train_class_names - val_class_names

    # This is an error because there are classes we clearly want to evaluate but we'll never
    # generate predictions for them.
    if len(val_not_in_train) != 0:
        raise ValueError(
            f"The following classes were in val but not train: {sorted(val_not_in_train)}"
        )

    # This case is slightly less severe because despite being unable to assess the quality of
    # these classes, they will still be present in the trained model's predictions.
    if len(train_not_in_val) != 0:
        print(
            f"Creating empty val folders for classes in train but not val: {sorted(train_not_in_val)}"
        )
        for name in train_not_in_val:
            (output_dir / "val" / name).mkdir()


if __name__ == "__main__":
    args = parse_args()

    main(**args.__dict__)
