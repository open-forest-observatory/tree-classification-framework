## Overview
The goal of this project is to support an end-to-end workflow for generating tree species prediction from individual views of the trees captured by drone images. It is developed as part of the [Open Forest Observatory](https://openforestobservatory.org/), which provides data and tools to support automated forest mapping and monitoring. This project serves as a light-weight connector between other Open Forest Observatory tools, including [geograypher](https://github.com/open-forest-observatory/geograypher), [tree-detection-framework](https://github.com/open-forest-observatory/tree-detection-framework), [tree-registration-and-matching](https://github.com/open-forest-observatory/tree-registration-and-matching), and our fork of [mmpretrain](https://github.com/open-forest-observatory/mmpretrain). There are two related workflows enabled by this project. The first is supporting training per-view species predictions models, using field reference information. The second is generating species predictions on unlabeled trees.

### Training workflow
The goal of this workflow is to train a per-view species prediction model starting with field reference species information. Trees are detected with the `tree-detection-framework` and then matched to field trees using the `tree-registration-and-matching` project. Then, using `geograypher`, the masks representing the locations of each tree as observed from each image rendered to the perspective of each image.

Within this project, the `chip_images.py` script saves out one image per view, per tree. This process optionally masks out background content. Importantly, the tree ID is preserved so these chips can be linked to the trees that generated them, which in turn can provide species information. Then, the `reorganize_chips.py` script is used to restructure this data into training and validation folders, organized per class. The `compute_summary_statistics.py` script is used to extract the metrics needed for model training from this data. Finally, an `mmpretrain` classification model can be trained using the [train.py](https://github.com/open-forest-observatory/mmpretrain/blob/main/tools/train.py) script.


### Inference workflow
The inference work starts with detected trees and geograypher renders of their locations onto the perspective of each image. Then, the `chip_images.py` script is used create one chip per view, which encodes the tree ID that generated the chip. Using a trained `mmpretrain` model, the [predict.py](https://github.com/open-forest-observatory/mmpretrain/blob/main/tools/predict.py) script is used to generate one class prediction per chip. Then the `assign_predictions_to_trees.py` script is used to map these prediction back to individual trees.


## Install
To install you must have `poetry` and `conda` installed. Then run the following commands.

```
conda create -n tree-classification-framework python=3.12
poetry install
```
In the future, use the created conda environment for all operations.

## Data
Example real data is provided on [Box](https://ucdavis.box.com/v/tree-classification-framework) (3.4GB). This contains products from three different sites: `001204_001205_0195`, `001419_001418_0042`, and `001438_001437_0003`. Most tools operate at the dataset level, except for the `reorganize_chips.py` which needs to be run on all datasets simaltanously. While only data in the `inputs` subfolder would be required in practice, results of running these tools are also included in `intermediate` and `final`, so any step can be run using these inputs.

The specific organization is listed below. Unless otherwise stated, within each folder listed below, there is a file or folder corresponding to each of the three datasets.

`inputs/`
- `images/`: images captured by the drone in a nested structure.
- `renders/`: per-image tree ID masks rendered by Geograypher. Each one corresponds to an image in `images`.
- `tree_tops/`: tree tops detected with a variable window filter from the CHM.
- `tree_crowns_matched/`: tree crowns detected from the CHM using the tree tops as seeds. These are then matched to point-based field reference data. In this product, only crowns which matched to a field tree are retained.
- `tree_crowns_unmatched/`: all detected tree crowns. This is a superset of the matched crowns and can be used for the prediction step.
- `train_val_split.csv`: a single file denoting whether each dataset should be included in the training or validation split
- `class-names-remapping.json`: a mapping from species codes to training classes

`intermediate/`
- `chips/`: views of each tree from each image, cropped and masked from the input images
- `predictions/`: per-chip class predictions, represented as .json
- `training_data/`: chips reorganized into train/val class folders and named sequentially

`final/`
- `tree_top_predictions/`: tree tops with per-tree classification predictions
- `tree_crown_predictions/`: tree crowns with per-tree classification predictions

## Scripts
### `chip_images.py`
[Geograypher](https://github.com/open-forest-observatory/geograypher) produces rendered masks per input image that identify the extent of each tree using a unique numerical ID. The goal of this step is to use these masks and the raw images to create a separate image for each view of each tree. This step optionally masks the non-focal tree content of the image with a solid color.

An example command using the example data is below.
```
python tree_classification_framework/chip_images.py \
  data/inputs/images/001438_001437_0003/ \
  data/inputs/renders/001438_001437_0003/ \
  data/intermediate/chips/001438_001437_0003 \
  --n-workers 4
```

### `reorganize_chips.py`
The previous chipping step creates one image per view of each tree. These are structured in the same way that the input imagery to photogrammetry was, with each original original image replaced by a folder full of chips. Each chip is named based on the tree's unique ID, which keys into a table of per-tree data provided as a `geopackage`. Each original dataset should have a folder of chips and a single tree-level data file. A `.csv` file is used to define multiple datasets to include in the reorganization, and for each dataset, whether it should be included in training or validation. Using a user-specified attribute of this data, the chips are reorganized into class-level folders within `train` or `val`. This format aligns with the standard [ImageFolder](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html) representation.

An example command using the example data is below.
```
python tree_classification_framework/reorganize_chips.py \
  data/intermediate/chips/ \
  data/inputs/tree_crowns_matched/ \
  data/inputs/train_val_split.csv \
  data/intermediate/training_data \
  --class-remap-file data/inputs/class-names-remapping.json \
  --filter-dead-trees --live-dead-attribute live_dead_prediction
```


### `compute_summary_statistics.py`
The reorganization step creates a folder of images formatted based on the [ImageFolder](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html) structure. This script computes several attributes that are helpful for downstream model training: the channel-wise mean and standard deviation, the number of classes, and their names. In most cases this script should be called on the training fold of the dataset.

An example command using the example data is below.
```
python tree_classification_framework/compute_summary_statistics.py \
  data/intermediate/training_data/train \
   --extension .png \
   --num-files 400
```

### `simulate_predictions.py`
In an inference workflow, the per-view chips from `chip_images` are classified using an computer vision model. This script serves as a drop-in replacement for an ML model and generates (semi)random predictions per chip in the same `.json` format expected by `assign_predictions_to_trees.py`. This can be helpful when trying to prototype downstream steps, when a real prediction model is not available or it is challenging to use it on the available computational resources. In this script, trees are assigned a label which can be either random or from a per-tree geospatial file. Per-view noise is added to simulate inaccurate predictions.

An example command using the example data is below.
```
python tree_classification_framework/simulate_predictions.py \
  data/intermediate/chips/001438_001437_0003/ \
  data/intermediate/predictions/001438_001437_0003.json \
  --reference-file data/inputs/tree_crowns_matched/001438_001437_0003.gpkg \
  --reference-attribute "species_code"
```

### `assign_predictions_to_trees.py`
The computer vision model (or `simulate_predictions.py`) produces chip-level predictions. Downstream tasks require a single label per tree, so this script implements a simple voting scheme to classify each tree based on the most commonly predicted class across all views. If a tie occurs between classes, it is broken randomly. This script also reports the number of views per tree and the fraction matching the prediction (modal class). The predictions are generated from chips which in turn a generated from the crowns. But this script can also optionally link these predictions back to the tree tops which were used to seed the crowns.

An example command using the example data is below
```
python tree_classification_framework/assign_predictions_to_trees.py \
  data/intermediate/predictions/001438_001437_0003.json \
  species_pred \
  data/inputs/tree_crowns_unmatched/001438_001437_0003.gpkg \
  data/final/tree_crown_predictions/001438_001437_0003.gpkg \
  --input-tree-tops-file data/inputs/tree_tops/001438_001437_0003.gpkg \
  --output-tree-tops-file data/final/tree_top_predictions/001438_001437_0003.gpkg
```
