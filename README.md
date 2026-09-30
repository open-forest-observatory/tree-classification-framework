## Install
To install you must have `poetry` and `conda` installed. Then run the following commands.

```
conda create -n tree-classification-framework python=3.12
poetry install
```
In the future, use the created conda environment for all operations.

## Scripts
`chip_images.py`: [Geograypher](https://github.com/open-forest-observatory/geograypher) produces rendered masks per input image that identify the extent of each tree using a unique numerical ID. The goal of this step is to use these masks and the raw images to create a separate image for each view of each tree. This step optionally masks the non-focal tree content of the image with a solid color.

`reorganize_chips.py`: The previous chipping step creates one image per view of each tree. These are structured in the same way that the input imagery to photogrammetry was, with each original original image replaced by a folder full of chips. Each chip is named based on the tree's unique ID, which keys into a table of per-tree data provided as a `geopackage`. Each original dataset should have a folder of chips and a single tree-level data file. A `.csv` file is used to define multiple datasets to include in the reorganization, and for each dataset, whether it should be included in training or validation. Using a user-specified attribute of this data, the chips are reorganized into class-level folders within `train` or `val`. This format aligns with the standard [ImageFolder](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html) representation.