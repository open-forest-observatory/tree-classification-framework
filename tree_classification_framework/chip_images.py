import json
import tempfile
import warnings
import shutil
from argparse import ArgumentParser, BooleanOptionalAction
from multiprocessing import Pool
from functools import partial
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from imageio.v2 import imread, imwrite
from PIL import Image
from rasterio import features
from rasterio.features import shapes
from shapely.affinity import translate
from tqdm import tqdm

# Filter out warning about saving without CRS, since the data represents pixel coords
warnings.filterwarnings("ignore", message="'crs' was not provided")


# Background masking configuration
MASK_BACKGROUND = True  # Whether to mask out background trees
MASK_BUFFER_PIXELS = 20  # Buffer zone around mask in pixels to retain
BACKGROUND_VALUE = (
    128,
    128,
    128,
)  # Value to set background pixels (0-255, recommend 128 for mid-gray)

# This value is what is used as the background value for the rendered masks.
# Be careful since IDs_to_labels must be overwritten to force 0 to be the background class during
# the rendering process.
RENDER_NULL_ID = 0

# Other parameters
BBOX_PADDING_RATIO = 0.02
# min edge length (height and width) to save
IMAGE_RES_MIN_SIZE = 50
# Any chip with both dimensions greater than this size will have a chance to be included
IMAGE_RES_SUFFICIENT_SIZE = 250
# How many chips to save out per tree
N_CHIPS_PER_TREE = 10
# Polygons smaller than this fraction of the largest polygon with the same ID are removed
FRAC_OF_MAX_SIZE = 0.5


def extract_shapes_from_mask(
    mask_path: str | Path,
    output_path: str | Path,
    render_null_ID: int = RENDER_NULL_ID,
    frac_of_max_size: float = FRAC_OF_MAX_SIZE,
):
    """
    Take a path to a one-channel image and extract the vector representations corresponding to the
    different unique values within the mask.

    Args:
        mask_path (str | Path): Path to a one-channel integer image, where unique IDs define the different trees
        output_path (str | Path): Where to save the vector results. Parent directory will be created if needed.
        render_null_ID (int, optional): The ID of the background content in the mask, which is not included. Defaults to RENDER_NULL_ID.
        frac_of_max_size (float, optional): Polygons with an area less than or equal to this fraction of the largest polygon with the same ID are removed. Defaults to FRAC_OF_MAX_SIZE.
    """
    mask_ids = imread(mask_path)  # load tif tree id mask
    mask_ids = np.squeeze(mask_ids)  # (H, W, 1) -> (H, W)

    # As far as I can tell, this just means there are no trees present and this caused a max-uint32
    # image to be saved out
    # See the issue here: https://github.com/open-forest-observatory/geograypher/issues/230
    if np.all(mask_ids == 2147483648):
        return

    # The background is all non-tree pixels
    individual_shapes = list(shapes(mask_ids, mask=mask_ids != render_null_ID))

    # No polygons, skip
    if len(individual_shapes) == 0:
        return

    # Extract the geometry from each shape along with its integer ID. `shapes` returns one Polygon
    # per connected region, so an ID split into multiple pieces will have multiple rows
    geometry_ids = [
        (shapely.geometry.shape(shape[0]), int(shape[1])) for shape in individual_shapes
    ]
    # Split into geometries and IDs
    geometry, ids = list(zip(*geometry_ids))
    # Create a geodataframe. Note, this data is not geospatial, but this is the easiest way abstract
    # working with vector data.
    shapes_gdf = gpd.GeoDataFrame({"geometry": geometry, "IDs": ids})

    # Store the area as an attribute for future use
    shapes_gdf["polygon_area"] = shapes_gdf.area
    # Find the max area per ID
    max_area_per_class = shapes_gdf[["polygon_area", "IDs"]].groupby("IDs").max()

    # Merge the area and max area by IDs
    shapes_gdf = shapes_gdf.join(max_area_per_class, on="IDs", rsuffix="_max")

    # Compute for each polygon what fraction of the max area for that ID it is
    shapes_gdf["frac_of_max"] = (
        shapes_gdf["polygon_area"] / shapes_gdf["polygon_area_max"]
    )
    # Remove the polygons which are less than the threshold fraction of the max for that ID
    shapes_gdf = shapes_gdf[shapes_gdf["frac_of_max"] > frac_of_max_size]
    # Remove the columns we no longer need
    shapes_gdf = shapes_gdf.drop(
        ["frac_of_max", "polygon_area", "polygon_area_max"], axis=1
    )
    # Dissolve by IDs to get one (potentially-multipolygon) entry per ID
    shapes_gdf = shapes_gdf.dissolve(by="IDs", as_index=False)

    shapes_gdf["filename"] = mask_path

    # Compute the minimum dimension per (multi)polygon
    width = shapes_gdf.bounds.maxx - shapes_gdf.bounds.minx
    height = shapes_gdf.bounds.maxy - shapes_gdf.bounds.miny
    shapes_gdf["min_dim"] = np.minimum(width, height)

    # Save out
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    shapes_gdf.to_file(output_path)


def save_chips(
    image_path: str | Path,
    shapes_path: str | Path,
    output_folder: str | Path,
    IDs_to_include: pd.Series,
    IDs_to_labels: dict,
    mask_background: bool = MASK_BACKGROUND,
    mask_buffer_pixels: int = MASK_BUFFER_PIXELS,
    background_value: tuple = BACKGROUND_VALUE,
    bbox_padding_ratio: float = BBOX_PADDING_RATIO,
):
    """
    Use the vector representation of the rendered mask to chip and save one image per tree.

    image_path (str | Path):
        Path to an RGB image, which will be chipped
    shapes_path (str | Path):
        A path to a dataframe of shapes representing the rendered trees, containing the "IDs" attribute
    output_folder (str | Path):
        Where to write all chips.
    IDs_to_include (pd.Series):
        A series containing which IDs to produce renders for.
    IDs_to_labels (dict):
        Mapping from integer values in the mask image to the filenames used for the output chips
    mask_background (bool, optional):
        Should the content outside of the geometry be set to a background value. Defaults to MASK_BACKGROUND.
    mask_buffer_pixels (int, optional):
        How many pixels to expand the geometry. Defaults to MASK_BUFFER_PIXELS.
    background_value (tuple, optional):
        The RGB color to use for the background if masking is applied. Defaults to BACKGROUND_VALUE.
    bbox_padding_ratio: (float, optional):
        The bounding box is this fraction larger on each edge than the mask that produces it. Defaults to BBOX_PADDING_RATIO.

    Raises:
        ValueError: If values in the mask image are not included in the IDs_to_labels keys, meaning they cannot be remapped
    """
    # Load the shapes
    shapes_gdf = gpd.read_file(shapes_path)
    # Subset to the IDs which should be written out
    shapes_gdf = shapes_gdf[shapes_gdf.IDs.isin(IDs_to_include)]

    # If there are no shapes to save then don't waste time loading the image
    if len(shapes_gdf) == 0:
        return

    # load image and convert to a numpy array for masking
    img_array = np.array(Image.open(image_path))

    # Check that all items can be remapped
    if not (shapes_gdf.IDs.isin(IDs_to_labels.keys())).all():
        un_mapped_values = list(
            set(list(shapes_gdf.IDs.unique())) - set(list(IDs_to_labels.keys()))
        )
        raise ValueError(
            f"Not all values could be remapped: {un_mapped_values} for image {image_path}"
        )
    # This cannot be done inplace in modern versions of pandas
    shapes_gdf.IDs = shapes_gdf.IDs.replace(IDs_to_labels)

    # Make the output folder
    Path(output_folder).mkdir(exist_ok=True, parents=True)

    # Compute the crop locations
    minx = shapes_gdf.geometry.bounds.minx
    miny = shapes_gdf.geometry.bounds.miny
    maxx = shapes_gdf.geometry.bounds.maxx
    maxy = shapes_gdf.geometry.bounds.maxy

    width = maxx - minx
    height = maxy - miny

    pad_width = width * bbox_padding_ratio
    pad_height = height * bbox_padding_ratio

    # padded coords for cropping
    left = minx - pad_width - mask_buffer_pixels
    top = miny - pad_height - mask_buffer_pixels
    right = maxx + pad_width + mask_buffer_pixels
    bottom = maxy + pad_height + mask_buffer_pixels

    # image shape (rows=height, cols=width)
    img_h, img_w = img_array.shape[:2]

    # integer pixel coordinates, clamped to image bounds
    shapes_gdf["crop_minx"] = np.maximum(0, np.floor(left)).astype(int)
    shapes_gdf["crop_miny"] = np.maximum(0, np.floor(top)).astype(int)
    shapes_gdf["crop_maxx"] = np.minimum(img_w, np.ceil(right)).astype(int)
    shapes_gdf["crop_maxy"] = np.minimum(img_h, np.ceil(bottom)).astype(int)

    # Buffering is only required if we're masking the background, but it's best to do it upfront
    # rather than on each iteration.
    if mask_background:
        # Catch warnings about invalid geometries
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # Expand the mask
            shapes_gdf.geometry = shapes_gdf.buffer(mask_buffer_pixels)

    # iterate over ids and save out each chip
    for _, row in shapes_gdf.iterrows():

        # extract crop
        crop = img_array[
            row.crop_miny : row.crop_maxy, row.crop_minx : row.crop_maxx
        ].copy()

        # Apply background masking if enabled
        if mask_background:
            # shift geometry into crop-local coordinates (use integer crop offsets)
            shifted_geometry = translate(
                row.geometry, xoff=-row.crop_minx, yoff=-row.crop_miny
            )

            # rasterize the shifted geometry to a mask (0 inside geometry, 1 outside)
            mask = features.rasterize(
                [(shifted_geometry, 0)],
                out_shape=(crop.shape[0], crop.shape[1]),
                fill=1,
                dtype="uint8",
            ).astype(bool)

            bg = np.array(background_value, dtype=crop.dtype)
            crop[mask] = bg

        # Create the output path
        output_path = Path(output_folder, f"{row.IDs}.png")

        # save cropped img
        imwrite(output_path, crop)


def subset_shapes(
    shapes, n_chips_per_tree, image_res_min_size, image_res_sufficient_size
):
    """
    Subset a GeoDataFrame of tree shapes to at most n_chips_per_tree chips per tree ID,
    filtering out chips that are too small to be useful.

    shapes (pd.DataFrame):
        A dataframe of shapes with "IDs" and "min_dim" attributes.
    n_chips_per_tree (int):
        Maximum number of chips to retain per tree ID.
    image_res_min_size (int):
        Minimum acceptable chip size in pixels. Chips smaller than this are excluded.
    image_res_sufficient_size (int):
        Chip size above which all chips are eligible for inclusion. The per-ID size
        threshold is never set higher than this value.

    Returns:
        pd.DataFrame: Filtered and sampled subset of the input shapes.
    """
    if n_chips_per_tree < 1:
        raise ValueError(f"n_chips_per_tree must be positive but is {n_chips_per_tree}")

    # Compute the minimum size per ID, by selecting the 2*n_chips_per_tree th highest size
    min_size_per_ID = shapes.groupby("IDs").apply(
        lambda x: x.nlargest(2 * n_chips_per_tree, "min_dim").iloc[-1]["min_dim"],
        include_groups=False,
    )
    # The min_size ensures that all chips are above a size that's feasible to generate a reasonable prediction on.
    # The sufficient_size means that all chips above this size should have a chance for inclusion,
    # so the minimum size should never be set higher than it.
    min_size_per_ID = min_size_per_ID.clip(
        image_res_min_size, image_res_sufficient_size
    )

    # Merge in the min size to the size per shapes
    shapes = shapes.merge(
        min_size_per_ID.rename("min_size_per_ID"), left_on="IDs", right_index=True
    )

    # Remove chips that are smaller than the threshold
    shapes = shapes[shapes["min_dim"] >= shapes["min_size_per_ID"]]
    # Select n_chips_per_tree from each ID or all, whichever is less
    shapes = (
        shapes.groupby("IDs")
        .apply(
            lambda x: x.sample(n=min(len(x), n_chips_per_tree)), include_groups=False
        )
        .reset_index(level=0)
        .reset_index(drop=True)
    )

    return shapes


def process_folder(
    images_folder: str | Path,
    renders_folder: str | Path,
    output_dir: str | Path,
    images_ext: str = ".JPG",
    renders_ext: str = ".tif",
    n_workers: int = 1,
    ensure_all_images_have_renders: bool = False,
    mask_background: bool = MASK_BACKGROUND,
    mask_buffer_pixels: int = MASK_BUFFER_PIXELS,
    background_value: tuple = BACKGROUND_VALUE,
    image_res_min_size: int = IMAGE_RES_MIN_SIZE,
    image_res_sufficient_size: int = IMAGE_RES_SUFFICIENT_SIZE,
    bbox_padding_ratio: float = BBOX_PADDING_RATIO,
    n_chips_per_tree: int = N_CHIPS_PER_TREE,
    frac_of_max_size: float = FRAC_OF_MAX_SIZE,
) -> None:
    """
    Chip every image in a folder based on a folder of mask images with a parallel structure, writing
    out the results in a parallel structure as the inputs. For more information, inspect the docstring
    of `chip_images`.
    """
    images_folder = Path(images_folder)
    renders_folder = Path(renders_folder)
    output_dir = Path(output_dir)

    # Check inputs
    image_files = sorted(images_folder.rglob(f"*{images_ext}"))
    render_files = sorted(renders_folder.rglob(f"*{renders_ext}"))

    images_stems = [f.relative_to(images_folder).with_suffix("") for f in image_files]
    renders_stems = [
        f.relative_to(renders_folder).with_suffix("") for f in render_files
    ]

    missing_images = set(renders_stems) - set(images_stems)
    # This is always a failure since these views cannot be chipped
    if len(missing_images) > 0:
        raise ValueError(
            f"{len(missing_images)} renders do not have a corresponding images. The first 10 are {list(missing_images)[:10]}"
        )

    # In some cases, only a subset of the views are rendered, for example with a spatial subset.
    if ensure_all_images_have_renders:
        additional_images = set(images_stems) - set(renders_stems)
        if len(additional_images) > 0:
            raise ValueError(
                f"{len(additional_images)} images do not have a corresponding renders. The first 10 are {list(additional_images)[:10]}"
            )

    # If all checks succeed and the output folder is present, delete it
    if output_dir.is_dir():
        shutil.rmtree(output_dir)

    # Where the vector representation of the masks is stored
    shapes_temp_dir = tempfile.TemporaryDirectory()

    # Create paths within the temp dir to store each file
    output_files = [
        Path(shapes_temp_dir.name, f.relative_to(renders_folder)).with_suffix(".gpkg")
        for f in render_files
    ]

    # Extract all vector representations of trees across all images
    with Pool(n_workers) as p:
        futures = [
            p.apply_async(
                extract_shapes_from_mask,
                (render_file, output_file, RENDER_NULL_ID, frac_of_max_size),
            )
            for render_file, output_file in zip(render_files, output_files)
        ]
        for f in tqdm(futures, desc="Extracting shapes from masks"):
            f.get()

    print("Determining a subset of chips to save")
    # Read only the attributes required to perform subsetting, since the geometry is memory intensive
    all_dimensions = []
    for f in Path(shapes_temp_dir.name).rglob("*.gpkg"):
        gdf = gpd.read_file(f)
        if len(gdf) > 0:
            all_dimensions.append(gdf[["filename", "min_dim", "IDs"]])
    if len(all_dimensions) == 0:
        raise ValueError(f"No trees were found in any of the masks in {renders_folder}")
    all_dimensions = pd.concat(all_dimensions, ignore_index=True)

    # Check how many top level folders there are, which is used to infer if this is a single or
    # paired mission case
    top_level_folder = np.array(
        [
            str(Path(f).relative_to(renders_folder).parts[0])
            for f in all_dimensions.filename
        ]
    )
    unique_folders = np.unique(top_level_folder)

    # Single mission case
    if len(unique_folders) == 1:
        all_dimensions = subset_shapes(
            all_dimensions,
            n_chips_per_tree,
            image_res_min_size,
            image_res_sufficient_size,
        )
    # Paired (oblique + nadir) missions
    elif len(unique_folders) == 2:
        # Apply the filtering procedure to the two top-level folders independently, which correspond
        # to the oblique and nadir missions
        all_dimensions_subsetted = []
        # Iterate over the folders corresponding to oblique and nadir, selected up to half of
        # n_chips_per_tree from each
        for unique_folder in unique_folders:
            all_dimensions_subsetted.append(
                subset_shapes(
                    all_dimensions[top_level_folder == unique_folder],
                    int(n_chips_per_tree / 2),
                    image_res_min_size,
                    image_res_sufficient_size,
                )
            )

        all_dimensions = pd.concat(all_dimensions_subsetted)
    else:
        raise ValueError(
            f"Expected one or two unique folders but got {len(unique_folders)}"
        )

    # Group the dimensions by filename to process each image independently
    dimensions_by_file = dict(tuple(all_dimensions.groupby("filename")))

    # Read IDs to labels
    with open(Path(renders_folder, "IDs_to_labels.json"), "r") as file_h:
        IDs_to_labels = json.load(file_h)
        IDs_to_labels = {int(k): v for k, v in IDs_to_labels.items()}

    # Create a partial function with all args that do not change across files
    partial_save_chips = partial(
        save_chips,
        IDs_to_labels=IDs_to_labels,
        mask_background=mask_background,
        mask_buffer_pixels=mask_buffer_pixels,
        background_value=background_value,
        bbox_padding_ratio=bbox_padding_ratio,
    )
    # Build args for parallel save_chips calls, using the fact that the images, shapes, and output
    # folders all have the same structure
    save_chips_args = [
        (
            (images_folder / Path(render_file).relative_to(renders_folder)).with_suffix(
                images_ext
            ),
            (
                shapes_temp_dir.name / Path(render_file).relative_to(renders_folder)
            ).with_suffix(".gpkg"),
            (output_dir / Path(render_file).relative_to(renders_folder)).with_suffix(
                ""
            ),
            dimensions_subset["IDs"],
        )
        for render_file, dimensions_subset in dimensions_by_file.items()
    ]

    # Save out the chips, parallelizing across files
    with Pool(n_workers) as p:
        futures = [p.apply_async(partial_save_chips, args) for args in save_chips_args]
        for f in tqdm(futures, desc="Saving out chips"):
            f.get()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "images_folder",
        type=Path,
        help="Path to a folder of input RGB images to be chipped",
    )
    parser.add_argument(
        "renders_folder",
        type=Path,
        help="Path to a folder of one-channel label images (.tif) representing the masks for each tree. This structure should exactly parallel the input images.",
    )
    parser.add_argument(
        "output_folder",
        type=Path,
        help="Where to saved the chipped data. The folder structure will parallel that of the input data, with one folder per original image. Each leaf folder will contain an image per tree.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="This process is highly parallelizable so you can run it multiprocessed",
    )
    parser.add_argument(
        "--ensure-all-images-have-renders",
        action="store_true",
        help="Fail if any images do not have a corresponding render",
    )
    parser.add_argument(
        "--mask-background",
        default=MASK_BACKGROUND,
        action=BooleanOptionalAction,
        help="Whether to mask out background pixels (default: %(default)s).",
    )
    parser.add_argument(
        "--mask-buffer-pixels",
        type=int,
        default=MASK_BUFFER_PIXELS,
        help="Buffer zone around mask in pixels to retain (default: %(default)s).",
    )
    parser.add_argument(
        "--background-value",
        type=int,
        nargs=3,
        default=list(BACKGROUND_VALUE),
        metavar=("R", "G", "B"),
        help="RGB value for background pixels, 0-255 (default: %(default)s).",
    )
    parser.add_argument(
        "--image-res-min-size",
        type=int,
        default=IMAGE_RES_MIN_SIZE,
        help="Minimum edge length (height and width) in pixels to save a chip (default: %(default)s).",
    )
    parser.add_argument(
        "--image-res-sufficient-size",
        type=int,
        default=IMAGE_RES_SUFFICIENT_SIZE,
        help="If the height and width of a chip are greater than this value, there will be a chance it will get saved. (default: %(default)s).",
    )
    parser.add_argument(
        "--bbox-padding-ratio",
        type=float,
        default=BBOX_PADDING_RATIO,
        help="The bounding box is this fraction larger on each edge than the mask that produces it. (default: %(default)s).",
    )
    parser.add_argument(
        "--n-chips-per-tree",
        type=int,
        default=N_CHIPS_PER_TREE,
        help="Save this many crops per tree",
    )
    parser.add_argument(
        "--frac-of-max-size",
        type=float,
        default=FRAC_OF_MAX_SIZE,
        help="Remove polygons with an area less than or equal to this fraction of the largest polygon with the same ID (default: %(default)s).",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Parse args
    args = parse_args()

    # Run
    process_folder(
        args.images_folder,
        args.renders_folder,
        args.output_folder,
        n_workers=args.n_workers,
        ensure_all_images_have_renders=args.ensure_all_images_have_renders,
        mask_background=args.mask_background,
        mask_buffer_pixels=args.mask_buffer_pixels,
        background_value=tuple(args.background_value),
        image_res_min_size=args.image_res_min_size,
        image_res_sufficient_size=args.image_res_sufficient_size,
        bbox_padding_ratio=args.bbox_padding_ratio,
        n_chips_per_tree=args.n_chips_per_tree,
        frac_of_max_size=args.frac_of_max_size,
    )
