"""
Saving frames from a .seq file into .tiff files
Cases:

1) Recorded in Kelvin with thresholds of 200 and 1000 degrees Celsius
Then we will record in Kelvin with noise clipping,
where the minimum value is 473.15 (200 degrees Celsius) and the maximum value is 1273.15 (1000 degrees Celsius)

2) Recorded in Celsius with thresholds of 0 and 500 degrees Celsius
Then we record in Celsius with noise clipping,
where the minimum value is 0 (Celsius) and the maximum value is 500 (Celsius)
"""
import os
import argparse
import shutil

import numpy as np
import tifffile as tiff

from tqdm import tqdm

from fe_tools.seq import Seq
from fe_tools.fff_tools import get_thermal_image


def main():
    args = parse_args()
    folder = args.input_file.split(".")[0]
    file_basename = os.path.basename(folder)
    make_empty_folder(folder)
    make_empty_folder(f"{folder}/fff_frames")
    make_empty_folder(f"{folder}/tiff_frames")

    print(f"Processing {args.input_file}...")
    for frame_id, frame_bytes in enumerate(tqdm(Seq(args.input_file))):
        # Write frame to .fff file
        fff_path = f"{folder}/fff_frames/{frame_id:04d}.fff"
        with open(fff_path, "wb") as fff_file:
            fff_file.write(frame_bytes)

        # Get thermal image from .fff file
        try:
            thermal_image = get_thermal_image(fff_path, is_celsius=args.is_celsius)
        except Exception as e:
            print(f"Error: {e}")
            continue

        # Save thermal frame as .tiff
        tiff_path = f"{folder}/tiff_frames/{file_basename}_{frame_id:04d}.tiff"
        # Trimming noises
        if not args.is_celsius:
            tiff_img = np.clip(thermal_image, args.min_thr + 273.15, args.max_thr + 273.15)
        else:
            tiff_img = np.clip(thermal_image, args.min_thr, args.max_thr)
        tiff.imwrite(tiff_path, tiff_img, photometric="minisblack")

    # Remove all .fff files
    # shutil.rmtree(f"{folder}/fff_frames")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Converts .seq file to .tiff files")
    parser.add_argument("input_file", type=str, help="Input .seq file")
    parser.add_argument("min_thr", type=float, help="Min temperature threshold of capturing")
    parser.add_argument("max_thr", type=float, help="Max temperature threshold of capturing")
    parser.add_argument(
            "--celsius",
            dest="is_celsius",
            action="store_true",
            help="Temperature in celsius in the .seq file",
            default=False
    )
    args = parser.parse_args()
    return args


def make_empty_folder(path: str) -> None:
    if os.path.exists(path):
        clear = input(f"{path} already exists. Clear it? (y/n): ")
        if clear == "y":
            shutil.rmtree(path)
        else:
            print("Exiting...")
            exit(1)
    os.makedirs(path)


if __name__ == "__main__":
    main()
