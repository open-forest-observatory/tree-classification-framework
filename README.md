## Install
To install you must have `poetry` and `conda` installed. Then run the following commands.

```
conda create -n tree-classification-framework python=3.12
poetry install
```
In the future, use the created conda environment for all operations.

## Scripts
`chip_images.py`: [Geograypher](https://github.com/open-forest-observatory/geograypher) produces rendered masks per input image that identify the extent of each tree using a unique numerical ID. The goal of this step is to use these masks and the raw images to create a separate image for each view of each tree. This step optionally masks the non-focal tree content of the image with a solid color.