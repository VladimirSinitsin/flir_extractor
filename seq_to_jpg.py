"""
Saving frames from a .seq file to .jpg files
Cases:

1) Recording in Kelvin with thresholds of 200 and 1000 degrees Celsius
Record jpg in grays. Where 0 is 200 (Celsius), 255 is 1000 (Celsius)

2) Recording in degrees Celsius with thresholds of 0 and 500 degrees Celsius
We record the jpg in grayscale. Where 0 is 0 (Celsius), 255 is 500 (Celsius)
"""
import os
import cv2
import shutil
import argparse
import numpy as np

from tqdm import tqdm

from fe_tools.seq import Seq
from fe_tools.fff_tools import get_thermal_image
from fe_tools.fff_tools import get_thermal_image_vis_gray


def main():
    args = parse_args()
    seq2jpg(args.input_file, args.min_thr, args.max_thr, args.is_celsius, args.is_debug)


def seq2jpg(seq_path: str, min_thr: float, max_thr: float, is_celsius: bool, is_debug: bool) -> None:
    """
    Saves frames from a .seq file into .jpg files
    :param seq_path: path to .seq file
    :param min_thr: min temperature threshold of capturing
    :param max_thr: max temperature threshold of capturing
    :param is_celsius: if temperature in .seq file is in celsius
    :param is_debug: create new folders for debug
    :return:
    """
    folder = seq_path.split(".")[0]
    file_basename = os.path.basename(folder)
    make_empty_folder(folder, is_debug)
    make_empty_folder(f"{folder}/fff_frames", is_debug)
    make_empty_folder(f"{folder}/jpg_frames", is_debug)

    seq_iterator = Seq(seq_path)
    fff_exists = len(os.listdir(f"{folder}/fff_frames")) == len(seq_iterator)
    print(f"Extracting JPG from {seq_path}...")
    for frame_id, frame_bytes in enumerate(tqdm(seq_iterator)):
        fff_path = f"{folder}/fff_frames/{frame_id:04d}.fff"
        if not fff_exists:
            # Write frame to .fff file
            with open(fff_path, "wb") as fff_file:
                fff_file.write(frame_bytes)

        # Get thermal image from .fff file
        try:
            thermal_image = get_thermal_image(fff_path, is_celsius=is_celsius)
            if frame_id == 0:
                ans = input(f"Temperature range is {np.min(thermal_image):.1f} - {np.max(thermal_image):.1f}. "
                            f"Do you want to change the unit of measurement? "
                            f"(current: {'celsius' if is_celsius else 'kelvin'})? [y/n]")
                if ans == "y":
                    is_celsius = not is_celsius
                    thermal_image = get_thermal_image(fff_path, is_celsius=is_celsius)
        except Exception as e:
            print(f"Error: {e}")
            continue

        gray_img = get_thermal_image_vis_gray(thermal_image, min_thr, max_thr)

        # Save thermal frame as .jpg
        jpg_path = f"{folder}/jpg_frames/{file_basename}_{frame_id:04d}.jpg"
        cv2.imwrite(jpg_path, gray_img)

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
    parser.add_argument(
            "--debug",
            dest="is_debug",
            action="store_true",
            help="Create new folders for debug",
            default=False
    )
    args = parser.parse_args()
    return args


def make_empty_folder(path: str, is_debug: bool) -> None:
    if not os.path.exists(path):
        os.makedirs(path)
    elif is_debug:
        shutil.rmtree(path)
        os.makedirs(path)


if __name__ == "__main__":
    main()
