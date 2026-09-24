import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
import argparse

import geopandas as gpd
import pandas as pd
import numpy as np

# Per-dataset folders each contain a "<dataset-id>_matched-trees.gpkg" and a "chips/" tree
# of "<unique_ID>.png" chips (multiple views per tree, one per source image subfolder).
GPKG_SUFFIX = "_matched-trees.gpkg"
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
ID_COLUMN = "unique_ID"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reorganize chips into a train/val folder structure for classification"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Input directory containing train/ and val/ subfolders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory to write reorganized chips to",
    )
    parser.add_argument(
        "--remap-file",
        type=Path,
        default=None,
        help="JSON file mapping original attribute values to final class names",
    )
    parser.add_argument(
        "--attribute-to-train-on",
        type=str,
        required=True,
        help="Column name in the geopackage to use as the class label",
    )
    parser.add_argument(
        "--filter-dead-trees",
        action="store_true",
        help="If set, drop trees predicted as dead based on the dead-trees-attribute column",
    )
    parser.add_argument(
        "--dead-trees-attribute",
        type=str,
        default="predicted_health_status",
        help="Column name in the geopackage indicating whether a tree is live or dead (default: predicted_health_status)",
    )
    return parser.parse_args()


def link_or_copy(src, dst):
    """Hardlink src -> dst, falling back to a copy if they are on different filesystems."""
    dst.parent.mkdir(exist_ok=True, parents=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main(
    input_dir,
    output_dir,
    remap_file,
    attribute,
    filter_dead_trees,
    dead_trees_attribute,
):
    # --- Load the class remapping (original attribute value -> final class name) ---
    # With no file provided, fall back to an identity mapping that keeps every observed class.
    if remap_file:
        with open(remap_file, "r") as f:
            class_remapping = {str(k): str(v) for k, v in json.load(f).items()}
        identity_mapping = False
        print(
            f"Loaded class remapping: {len(class_remapping)} attribute values -> "
            f"{len(set(class_remapping.values()))} output classes"
        )
    else:
        class_remapping = {}
        identity_mapping = True
        print("No class remapping file provided; keeping every class unchanged")

    if not (input_dir / "train").is_dir() or not (input_dir / "val").is_dir():
        raise ValueError(
            f"Both train and val subfolders of {input_dir} must be present"
        )

    for train_val in ("train", "val"):
        # This saves information about the restructuring
        reorganization_summary = []

        # Create the subfolder path that holds all datasets for either train or val
        input_dir_train_val = input_dir / train_val
        # --- List the per-dataset folders (those holding a *_matched-trees.gpkg) ---
        dataset_dirs = sorted(
            d
            for d in input_dir_train_val.iterdir()
            if d.is_dir() and any(d.glob(f"*{GPKG_SUFFIX}"))
        )
        if not dataset_dirs:
            raise ValueError(
                f"No dataset folders containing a *{GPKG_SUFFIX} file were found under {input_dir_train_val}"
            )
        print(
            f"Found {len(dataset_dirs)} dataset folder(s): {[d.name for d in dataset_dirs]}"
        )

        # --- Per-class counter: how many chips have been written for each class so far ---
        class_counters = defaultdict(int)
        skipped_unmapped = defaultdict(int)  # raw attribute value -> chips skipped
        chips_without_tree = 0  # chip whose unique_ID is absent from the geopackage

        for dataset_dir in dataset_dirs:
            gpkg_path = sorted(dataset_dir.glob(f"*{GPKG_SUFFIX}"))[0]
            chips_dir = dataset_dir / "chips"
            if not chips_dir.is_dir():
                print(f"[{dataset_dir.name}] no chips/ directory; skipping")
                continue

            # 1. Load the geopackage and remap the class attribute for each tree
            trees = gpd.read_file(gpkg_path)
            # Drop rows which are not predicted as live if requested
            if filter_dead_trees:
                n_total = len(trees)
                if dead_trees_attribute not in trees.columns:
                    raise ValueError(
                        f"[{dataset_dir.name}] column '{dead_trees_attribute}' not found in {gpkg_path.name}; "
                        f"available columns: {list(trees.columns)}"
                    )
                live_trees = trees[dead_trees_attribute] == "Live"
                dead_tree_IDs = trees.loc[~live_trees, ID_COLUMN].tolist()
                trees = trees[live_trees]

                print(
                    f"Dropped {n_total - len(trees)} of {n_total} trees predicted as dead from {dataset_dir.name}"
                )
            else:
                dead_tree_IDs = []

            for required in (ID_COLUMN, attribute):
                if required not in trees.columns:
                    raise ValueError(
                        f"[{dataset_dir.name}] column '{required}' not found in {gpkg_path.name}; "
                        f"available columns: {list(trees.columns)}"
                    )
            trees[ID_COLUMN] = trees[ID_COLUMN].astype(str)

            raw_by_id = dict(zip(trees[ID_COLUMN], trees[attribute]))
            class_by_id = {}
            for unique_id, raw_class in raw_by_id.items():
                if pd.isna(raw_class):
                    continue
                raw_class = str(raw_class)
                if identity_mapping:
                    class_by_id[unique_id] = raw_class
                elif raw_class in class_remapping:
                    class_by_id[unique_id] = class_remapping[raw_class]
                # else: attribute value absent from the remapping file -> drop this tree

            # 2. Hardlink every chip to the output folder for its remapped class
            n_linked = 0
            for chip_path in sorted(chips_dir.rglob("*")):
                if (
                    not chip_path.is_file()
                    or chip_path.suffix.lower() not in IMAGE_EXTS
                ):
                    continue
                unique_id = chip_path.stem
                output_class = class_by_id.get(unique_id)
                if output_class is None:
                    if unique_id in raw_by_id:
                        skipped_unmapped[str(raw_by_id[unique_id])] += 1
                    elif unique_id in dead_tree_IDs:
                        # This tree was dropped because it was predicted as dead
                        pass
                    else:
                        raise ValueError(
                            f"Unique ID {unique_id} not found in {gpkg_path.name}"
                        )
                    continue

                class_dir = output_dir / train_val / output_class
                dst = (
                    class_dir
                    / f"{class_counters[output_class]:06d}{chip_path.suffix.lower()}"
                )
                link_or_copy(chip_path, dst)
                class_counters[output_class] += 1
                n_linked += 1

                # Add an entry to the list which tracks the reorganization
                reorganization_summary.append(
                    {
                        "image_path": str(dst),
                        "class": output_class,
                        "tree_id": unique_id,
                        "dataset_id": dataset_dir.name,
                    }
                )

            print(
                f"[{dataset_dir.name}] linked {n_linked} chips "
                f"from {len(class_by_id)} matched trees"
            )

        # --- Summary ---
        total = sum(class_counters.values())
        print(f"\n=== Reorganization summary ({train_val})===")
        for output_class in sorted(class_counters):
            print(f"  {output_class}: {class_counters[output_class]} chips")
        print(f"  TOTAL: {total} chips across {len(class_counters)} class(es)")
        if skipped_unmapped:
            print(
                "Chips skipped (attribute value null or absent from the remapping file):"
            )
            for raw, count in sorted(skipped_unmapped.items(), key=lambda kv: -kv[1]):
                print(f"  {raw!r}: {count}")
        if chips_without_tree:
            print(
                f"Chips skipped (unique_ID not present in the geopackage): {chips_without_tree}"
            )

        if total == 0:
            raise ValueError(
                "No chips were written; check ATTRIBUTE_TO_TRAIN_ON and the remapping file"
            )

        # Save out the summary file
        reorganization_summary = pd.DataFrame(reorganization_summary)
        reorganization_summary_path = Path(
            output_dir, f"{train_val}_reorganization_summary.csv"
        )
        reorganization_summary.to_csv(reorganization_summary_path)

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
