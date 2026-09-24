import json
import tempfile
import warnings
from argparse import ArgumentParser, BooleanOptionalAction
from functools import partial
from multiprocessing import Pool
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


def extract_shapes_from_mask(
    mask_path: str,
    output_path: str,
    render_null_ID: int = RENDER_NULL_ID,
):
    """
    Take a path to a one-channel image and extract the vector representations corresponding to the
    different unique values within the mask.

    Args:
        mask_path (str): Path to a one-channel integer image, where unique IDs define the different trees
        output_path (str): Where to save the vector results. Parent directory will be created if needed.
        render_null_ID (int, optional): The ID of the background content in the mask, which is not included. Defaults to RENDER_NULL_ID.
    """
    mask_ids = imread(mask_path)  # load tif tree id mask
    mask_ids = np.squeeze(mask_ids)  # (H, W, 1) -> (H, W)

    if mask_ids.dtype == np.uint32:
        # Indicates a mallformed image in the current experiments
        return

    # The background is all non-tree pixels
    individual_shapes = list(shapes(mask_ids, mask=mask_ids != render_null_ID))

    # No polygons, skip
    if len(individual_shapes) == 0:
        return

    # Extract the potentially-multiple polygons from each shape along with the original value,
    # now encoded as a zero-padded string
    polys = [
        (shapely.Polygon(poly), int(shape[1]))
        for shape in individual_shapes
        for poly in shape[0]["coordinates"]
    ]
    # Split into geometries and IDs and build a geodataframe
    geometry, ids = list(zip(*polys))
    # Create a geodataframe. Note, this data is not geospatial, but this is the easiest way to work
    # abstract working with vector data.
    shapes_gdf = gpd.GeoDataFrame({"geometry": geometry, "IDs": ids})

    # Merge by ID, forming multipolygons as needed
    shapes_gdf = shapes_gdf.dissolve("IDs", as_index=False)

    shapes_gdf["filename"] = mask_path

    # Compute the minimum dimension per chip
    width = shapes_gdf.bounds.maxx - shapes_gdf.bounds.minx
    height = shapes_gdf.bounds.maxy - shapes_gdf.bounds.miny
    min_dim = np.minimum(width, height)
    shapes_gdf["min_dim"] = min_dim

    # Save out
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    shapes_gdf.to_file(output_path)


def save_chips(
    image_path: str,
    shapes_path: str,
    IDs_to_include: pd.Series,
    output_folder: str,
    IDs_to_labels: dict,
    mask_background: bool = MASK_BACKGROUND,
    mask_buffer_pixels: int = MASK_BUFFER_PIXELS,
    background_value: tuple = BACKGROUND_VALUE,
):
    """
    Use the vector representation of the rendered mask to chip and save one image per tree.

    image_path (str):
        Path to an RGB image, which will be chipped
    shapes_path (str):
        A path to a dataframe of shapes representing the rendered trees, containing the
        "polygon_area" and "IDs" attributes
    IDs_to_include (pd.Series):
        A series containing which IDs to produce renders for.
    output_folder (str):
        Where to write all chips.
    IDs_to_labels (dict):
        Mapping from integer values in the mask image to the filenames used for the output chips
    mask_background (bool, optional):
        Should the content outside of the geometry be set to a background value. Defaults to MASK_BACKGROUND.
    mask_buffer_pixels (int, optional):
        How many pixels to expand the geometry. Defaults to MASK_BUFFER_PIXELS.
    background_value (tuple, optional):
        The RGB color to use for the background if masking is applied. Defaults to BACKGROUND_VALUE.

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

    # load image
    img = Image.open(image_path)
    # Convert to numpy array for masking
    img_array = np.array(img) if mask_background else None

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
    shapes_gdf = shapes_gdf[shapes_gdf["frac_of_max"] > 0.5]
    # Remove the columns we no longer need
    shapes_gdf.drop(
        ["frac_of_max", "polygon_area", "polygon_area_max"], axis=1, inplace=True
    )

    # This cannot be done inplace in modern versions of pandas
    shapes_gdf.IDs = shapes_gdf.IDs.replace(IDs_to_labels)
    # Check that all items were remapped
    if not (shapes_gdf.IDs.isin(IDs_to_labels.values())).all():
        un_mapped_values = list(
            set(list(shapes_gdf.IDs.unique())) - set(list(IDs_to_labels.values()))
        )
        raise ValueError(f"Not all values were remapped: {un_mapped_values}")

    # Make the output folder
    Path(output_folder).mkdir(exist_ok=True, parents=True)

    # Compute the crop locations
    minx = shapes_gdf.geometry.bounds.minx
    miny = shapes_gdf.geometry.bounds.miny
    maxx = shapes_gdf.geometry.bounds.maxx
    maxy = shapes_gdf.geometry.bounds.maxy

    width = maxx - minx
    height = maxy - miny

    pad_width = width * BBOX_PADDING_RATIO
    pad_height = height * BBOX_PADDING_RATIO

    # padded coords for cropping
    # Don't inflate by the buffering amount if no masking is applied
    left = minx - pad_width - (mask_buffer_pixels if mask_background else 0)
    top = miny - pad_height - (mask_buffer_pixels if mask_background else 0)
    right = maxx + pad_width + (mask_buffer_pixels if mask_background else 0)
    bottom = maxy + pad_height + (mask_buffer_pixels if mask_background else 0)

    # image shape (rows=height, cols=width)
    img_h, img_w = img_array.shape[:2]

    # integer pixel coordinates, clamped to image bounds
    shapes_gdf["crop_minx"] = np.maximum(0, np.floor(left)).astype(int)
    shapes_gdf["crop_miny"] = np.maximum(0, np.floor(top)).astype(int)
    shapes_gdf["crop_maxx"] = np.minimum(img_w, np.ceil(right)).astype(int)
    shapes_gdf["crop_maxy"] = np.minimum(img_h, np.ceil(bottom)).astype(int)

    # Buffering is only required if we're masking the background, but it's best to do it upfront
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

    shapes (gpd.GeoDataFrame):
        A dataframe of shapes with "IDs" and "min_dim" attributes.
    n_chips_per_tree (int):
        Maximum number of chips to retain per tree ID.
    image_res_min_size (int):
        Minimum acceptable chip size in pixels. Chips smaller than this are excluded.
    image_res_sufficient_size (int):
        Chip size above which all chips are eligible for inclusion. The per-ID size
        threshold is never set higher than this value.

    Returns:
        gpd.GeoDataFrame: Filtered and sampled subset of the input shapes.
    """
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
    images_folder,
    renders_folder,
    output_dir,
    images_ext=".JPG",
    renders_ext=".tif",
    n_workers=1,
    ensure_all_images_have_renders=False,
    mask_background: bool = MASK_BACKGROUND,
    mask_buffer_pixels: int = MASK_BUFFER_PIXELS,
    background_value: tuple = BACKGROUND_VALUE,
    image_res_min_size: int = IMAGE_RES_MIN_SIZE,
    image_res_sufficient_size=IMAGE_RES_SUFFICIENT_SIZE,
    n_chips_per_tree=10,
) -> tuple:
    """
    Chip every image in a folder based on a folder of mask images with a parellel structure, writing
    out the results in a parellel structure as the inputs. For more information, inspect the docstring
    of `chip_images`.
    """
    images_folder = Path(images_folder)
    renders_folder = Path(renders_folder)

    image_files = sorted(images_folder.rglob(f"*{images_ext}"))
    render_files = sorted(renders_folder.rglob(f"*{renders_ext}"))

    images_stems = [f.relative_to(images_folder).with_suffix("") for f in image_files]
    renders_stems = [
        f.relative_to(renders_folder).with_suffix("") for f in render_files
    ]

    missing_images = set(renders_stems) - set(images_stems)
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

    # Where the vector representation of the masks is stored
    shapes_dir = tempfile.TemporaryDirectory()

    # Create paths within the temp dir to store each file
    output_files = [
        Path(shapes_dir.name, f.relative_to(renders_folder)).with_suffix(".gpkg")
        for f in render_files
    ]

    # Extract all vector representations of trees across all images
    with Pool(n_workers) as p:
        futures = [
            p.apply_async(extract_shapes_from_mask, args)
            for args in zip(render_files, output_files)
        ]
        for f in tqdm(futures, desc="Extracting shapes from masks"):
            f.get()

    print("Determining a subset of chips to save")
    # Read only the attributes required to perform subsetting, since the geometry is memory intensive
    all_dimensions = []
    for f in Path(shapes_dir.name).rglob("*.gpkg"):
        gdf = gpd.read_file(f)
        if len(gdf) > 0:
            all_dimensions.append(gdf[["filename", "min_dim", "IDs"]])
    all_dimensions = pd.concat(all_dimensions, ignore_index=True)

    # Apply the filtering proceedure to the two top-level folders independently, which correspond
    # to the oblique and nadir missions
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

    # Build args for parallel save_chips calls
    # TODO consider a partial function
    save_chips_args = [
        (
            str(
                Path(
                    images_folder,
                    Path(render_file).relative_to(renders_folder),
                ).with_suffix(images_ext)
            ),
            str(
                Path(
                    shapes_dir.name,
                    Path(render_file).relative_to(renders_folder),
                ).with_suffix(".gpkg")
            ),
            dimensions_subset["IDs"],
            str(
                Path(
                    output_dir,
                    Path(render_file).relative_to(renders_folder).with_suffix(""),
                )
            ),
            IDs_to_labels,
            mask_background,
            mask_buffer_pixels,
            background_value,
        )
        for render_file, dimensions_subset in dimensions_by_file.items()
    ]

    # Save out the chips, parallelizing across files
    with Pool(n_workers) as p:
        futures = [p.apply_async(save_chips, args) for args in save_chips_args]
        for f in tqdm(futures, desc="Saving out chips"):
            f.get()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("images_folder")
    parser.add_argument("renders_folder")
    parser.add_argument("output_folder")
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--ensure-all-images-have-renders", action="store_true")
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
        help="If the height and width of a chip are greater than this value, ther will be a chance it will get saved. (default: %(default)s).",
    )
    parser.add_argument(
        "--n-chips-per-tree",
        type=int,
        default=N_CHIPS_PER_TREE,
        help="Save this many crops per tree",
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
        n_chips_per_tree=args.n_chips_per_tree,
    )
