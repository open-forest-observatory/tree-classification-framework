import argparse
import json
import sys
from pathlib import Path

import numpy as np
from imageio.v3 import imread
from numpy.random import choice
from tqdm import tqdm


def compute_summary_statistics(
    dataset_folder: Path | str, extension: str = "", num_files: int | None = None
) -> tuple[list, list]:
    """Compute the channel-wise mean and standard deviation of a folder of images

    Args:
        dataset_folder (Path | str):
            Path to a folder of images. This will be searched recursively.
        extension (str, optional):
            Only include files with this extension. Defaults to "", which includes all files.
        num_files (int, optional):
            Subset the images to this number of random files for computational tractability. If
            unset, all images will be used. Defaults to None.

    Returns:
        list: channel-wise means
        list: channel-wise standard deviations
    """
    files = [x for x in Path(dataset_folder).rglob("*" + extension) if x.is_file()]

    if len(files) == 0:
        raise ValueError(
            f"No files in {dataset_folder}"
            + ("" if extension == "" else f" with extension {extension}")
        )

    if num_files is not None:
        files = choice(files, min(num_files, len(files)))

    imgs = [imread(x) for x in tqdm(files)]
    # Deal with different size images
    imgs = [np.reshape(img, (-1, 3)) for img in imgs]
    imgs = np.concatenate(imgs, axis=0)  # Concatenate vertically

    means = np.mean(imgs, axis=0).tolist()
    stds = np.std(imgs, axis=0).tolist()

    return means, stds


def main(
    dataset_folder: Path | str,
    extension: str = "",
    num_files: int | None = None,
    output_file: Path | None = None,
):
    """Compute the channel-wise summary statistics and class names and the number thereof.

    Args:
        dataset_folder (Path | str):
            Path to a folder of images. This will be searched recursively.
        extension (str, optional):
            see `compute_summary_statistics
        num_files (int, optional):
            see `compute_summary_statistics
        output_file (Path, optional)
            if provided, write the summary statistics to this file. Otherwise they are printed. Defaults to None.

    Writes to a file or prints to stdout the summary as a json-formatted string
    """
    # It's critical that the classes are sorted
    classes = sorted([p.name for p in list(Path(dataset_folder).glob("*"))])
    n_classes = len(classes)

    means, stds = compute_summary_statistics(
        dataset_folder, extension=extension, num_files=num_files
    )

    output_dict = {
        "classes": classes,
        "n_classes": n_classes,
        "means": means,
        "stds": stds,
    }

    # Print or write the output
    if output_file is None:
        json.dump(output_dict, sys.stdout)
    else:
        output_file.parent.mkdir(exist_ok=True, parents=True)
        with open(output_file, "w") as output_file_h:
            json.dump(output_dict, output_file_h, sort_keys=True, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(
        "This script computes summary statistics of a folder of images, already formatted for classification model training. This includes the channel-wise mean, standard deviation, list of classes and the number of classes. The data is returned in a json representation for easy downstream use."
    )
    parser.add_argument(
        "dataset_folder",
        type=Path,
        help="The input folder to compute the channel-wise mean and std of, as well as list the classes of. This is frequently the training folder.",
    )
    parser.add_argument(
        "--extension",
        type=str,
        help="Only search for this extension when computing the summary statistics. If unset, all files will be included.",
        default="",
    )
    parser.add_argument(
        "--num-files",
        type=int,
        help="Subset to this many random files prior to computing statistics.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="If provided, write the results to this .json file rather than printing them.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    main(args.dataset_folder, args.extension, args.num_files, args.output_file)
