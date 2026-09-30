## Overview

## Install
To install you must have `poetry` and `conda` installed. Then run the following commands.

```
conda create -n tree-classification-framework python=3.12
poetry install
```
In the future, use the created conda environment for all operations.

## Data

## Scripts
### `chip_images.py`
[Geograypher](https://github.com/open-forest-observatory/geograypher) produces rendered masks per input image that identify the extent of each tree using a unique numerical ID. The goal of this step is to use these masks and the raw images to create a separate image for each view of each tree. This step optionally masks the non-focal tree content of the image with a solid color.

An example command using the example data is below.
```
python tree_classification_framework/reorganize_chips.py \
  data/chips/ \
  data/tree_metadata/ \
  data/train_val_split.csv \
  data/training_data \
  --class-remap-file data/class-names-remapping.json \
  --filter-dead --live-dead-attribute live_dead_prediction
```

### `reorganize_chips.py`
The previous chipping step creates one image per view of each tree. These are structured in the same way that the input imagery to photogrammetry was, with each original original image replaced by a folder full of chips. Each chip is named based on the tree's unique ID, which keys into a table of per-tree data provided as a `geopackage`. Each original dataset should have a folder of chips and a single tree-level data file. A `.csv` file is used to define multiple datasets to include in the reorganization, and for each dataset, whether it should be included in training or validation. Using a user-specified attribute of this data, the chips are reorganized into class-level folders within `train` or `val`. This format aligns with the standard [ImageFolder](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html) representation.

An example command using the example data is below.
```
python tree_classification_framework/compute_summary_statistics.py \
  data/training_data/train \
   --extension .png \
   --num-files 100
```


### `compute_summary_statistics.py`
The reorganization step creates a folder of images formatted based on the [ImageFolder](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html) structure. This script computes several attributes that are helpful for downstream model training: the channel-wise mean and standard deviation, the number of classes, and their names. In most cases this script should be called on the training fold of the dataset.

An example command using the example data is below.
```
python tree_classification_framework/simulate_predictions.py \
  data/chips/001438_001437_0003/ \
  data/predictions/001438_001437_0003.json \
  --reference-file data/tree_crowns_matched/001438_001437_0003.gpkg \
  --reference-attribute "species_code"
```

### `simulate_predictions.py`
In an inference workflow, the per-view chips from `chip_images` are classified using an computer vision model. This script serves as a drop-in replacement for an ML model and generates (semi)random predictions per chip in the same `.json` format expected by `assign_predictions_to_trees.py`. This can be helpful when trying to prototype downstream steps, when a real prediction model is not available or it is challenging to use it on the available computational resources. In this script, trees are assigned a label which can be either random or from a per-tree geospatial file. Per-view noise is added to simulate inaccurate predictions.

An example command using the example data is below.
```
python tree_classification_framework/simulate_predictions.py \
  data/chips/001438_001437_0003/ \
  data/predictions/001438_001437_0003.json \
  --reference-file data/tree_crowns_matched/001438_001437_0003.gpkg \
  --reference-attribute "species_code"
```

### `assign_predictions_to_trees.py`
The computer vision model (or `simulate_predictions.py`) produces chip-level predictions. Downstream tasks require a single label per tree, so this script implements a simple voting scheme to classify each tree based on the most commonly predicted class across all views. If a tie occurs between classes, it is broken randomly. This script also reports the number of views per tree and the fraction matching the prediction (modal class). The predictions are generated from chips which in turn a generated from the crowns. But this script also links these predictions back to the tree tops which were used to seed the crowns.

An example command using the example data is below
```
python tree_classification_framework/assign_predictions_to_trees.py \
  data/predictions/001438_001437_0003.json species_pred data/tree_crowns/001438_001437_0003.gpkg \
  data/tree_tops/001438_001437_0003.gpkg data/tree_crown_predictions/001438_001437_0003.gpkg  \
  data/tree_top_predictions/001438_001437_0003.gpkg
```
