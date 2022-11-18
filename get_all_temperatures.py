import os
from typing import Tuple

import shutil
import numpy as np

from glob import glob
from tqdm import tqdm

from fe_tools.seq import Seq
from fe_tools.fff_tools import get_thermal_image


def main():
    main_dir = "/data/sinitsin/erg_kz/1_day"
    seq_files = sorted(glob(f"{main_dir}/*.seq"))
    if len(seq_files) == 0:
        main_dir = "erg_kz"
        seq_files = sorted(glob(f"{main_dir}/*.seq"))

    global_min, global_max = np.inf, -np.inf
    for seq_id, seq_path in enumerate(seq_files):
        print(f"Extracting {seq_path.split('/')[-1]} ({seq_id + 1}/{len(seq_files)}):")
        min, max = extract_seq_gray(seq_path)
        global_min = min if min < global_min else global_min
        global_max = max if max > global_max else global_max
    print(f"{global_min=:.1f}, {global_max=:.1f}")


def extract_seq_gray(seq_path: str) -> Tuple[int, int]:
    folder = os.path.join(os.path.dirname(seq_path), "converted", os.path.basename(seq_path).split(".")[0])
    file_basename = os.path.basename(folder)
    make_empty_folder(folder)
    make_empty_folder(f"{folder}/fff_frames")

    min, max = np.inf, -np.inf
    for frame_id, frame_bytes in enumerate(tqdm(Seq(seq_path))):
        # Write frame to .fff file
        fff_path = f"{folder}/fff_frames/{frame_id:04d}.fff"
        with open(fff_path, "wb") as fff_file:
            fff_file.write(frame_bytes)

        # Get thermal image from .fff file
        try:
            thermal_image = get_thermal_image(fff_path, is_celsius=False)
        except Exception as e:
            print(f"Error: {e}")
            continue
        frame_min, frame_max = np.min(thermal_image), np.max(thermal_image)
        min = frame_min if frame_min < min else min
        max = frame_max if frame_max > max else max

    shutil.rmtree(folder)
    print(f"{file_basename} {min=:.1f}, {max=:.1f}")
    return min, max


def make_empty_folder(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


if __name__ == "__main__":
    main()
