import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from imageio import imread
from numpy.random import choice
from tqdm import tqdm


def compute_summary_statistics(
    images, num_files=50, savepath=None, subsample_step=100, extension=""
):
    files = [x for x in Path(images).rglob("*" + extension) if x.is_file()]
    files = choice(files, min(num_files, len(files)))

    imgs = [imread(x) for x in tqdm(files)]
    # Deal with different size images
    imgs = [np.reshape(img, (-1, 3)) for img in imgs]
    imgs = np.concatenate(imgs, axis=0)  # Concatenate vertically
    # Subsample
    imgs = imgs[np.arange(0, imgs.shape[0], subsample_step).astype(int)]
    print(imgs.shape)
    mean = np.mean(imgs, axis=0)
    std = np.std(imgs, axis=0)

    means, stds = mean.tolist(), std.tolist()
    if savepath is not None:
        with open(savepath, "w") as outfile_h:
            outfile_h.write(f"mean per channel: {means}\n")
            outfile_h.write(f"std per channel: {stds}\n")
    return means, stds


train_folder = Path("{{inputs.parameters.dataset-folder}}", "train")
# It's critical that the classes are sorted
classes = sorted([p.name for p in list(train_folder.glob("*"))])
n_classes = len(classes)

# Suppress printing so the final output is json-parseable
with redirect_stdout(open(os.devnull, "w")):
    means, stds = compute_summary_statistics(train_folder)

output_dict = {"classes": classes, "n_classes": n_classes, "means": means, "stds": stds}

json.dump(output_dict, sys.stdout)
