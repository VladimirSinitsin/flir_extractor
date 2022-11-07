import os
import cv2
import shutil
import numpy as np
import tifffile as tiff

from tqdm import tqdm

from fe_tools.seq import Seq
from fe_tools.fff_tools import get_thermal_image
from fe_tools.fff_tools import get_thermal_image_vis_gray


def main():
    main_dir = "/data/sinitsin/erg_kz/1_day"
    seq_id = 0
    seq_files = sorted(os.listdir(main_dir))
    for seq_file in seq_files:
        if seq_file.endswith(".seq"):
            seq_id += 1
            print(f"Extracting {seq_file} ({seq_id}/{len(seq_files)}):")
            extract_seq_gray(f"{main_dir}/{seq_file}")


def extract_seq_gray(seq_path: str) -> None:
    folder = seq_path.split(".")[0]
    file_basename = os.path.basename(folder)
    make_empty_folder(folder)
    make_empty_folder(f"{folder}/fff_frames")
    # make_empty_folder(f"{folder}/thermal_frames")

    thermal_tensor = []
    for frame_id, frame_bytes in enumerate(tqdm(Seq(seq_path))):
        # Write frame to .fff file
        fff_path = f"{folder}/fff_frames/{frame_id:04d}.fff"
        with open(fff_path, "wb") as fff_file:
            fff_file.write(frame_bytes)

        # Get thermal image from .fff file
        try:
            thermal_image = get_thermal_image(fff_path)
        except Exception as e:
            print(f"Error: {e}")
            continue
        # os.remove(fff_path)

        # Add thermal image to tensor (for writing to .npz or .npy)
        thermal_tensor.append(thermal_image)

        # Save thermal frame as .jpg
        jpg_path = f"{folder}/{file_basename}_{frame_id:04d}.jpg"
        cv2.imwrite(jpg_path, get_thermal_image_vis_gray(thermal_image))

        # Save thermal frame as .tif
        tiff_path = f"{folder}/{file_basename}_{frame_id:04d}.tiff"
        tiff.imwrite(tiff_path, thermal_image, photometric="minisblack")

    # Remove all .fff files
    shutil.rmtree(f"{folder}/fff_frames")
    # Save .npz or .npy
    thermal_tensor = np.array(thermal_tensor)
    print(f"Saving thermal tensor {thermal_tensor.shape} ...")
    # np.save(f"{folder}/thermal.npy", thermal_tensor)
    np.savez_compressed(f"{folder}/{file_basename}.npz", thermal_tensor)
    print("Complete!")


def make_empty_folder(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


if __name__ == "__main__":
    main()
