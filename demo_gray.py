import os
import cv2
import shutil
import numpy as np

from tqdm import tqdm

from fe_tools.seq import Seq
from fe_tools.fff_tools import get_thermal_image
from fe_tools.fff_tools import get_thermal_image_vis_gray


def main():
    main_dir = "erg_kz/1_day"
    for seq_file in os.listdir(main_dir):
        if seq_file.endswith(".seq"):
            print(f"Ectracting {seq_file}:")
            extract_seq_gray(f"{main_dir}/{seq_file}")


def extract_seq_gray(seq_path: str) -> None:
    folder = seq_path.split(".")[0]
    make_empty_folder(folder)
    make_empty_folder(f"{folder}/fff_frames")
    make_empty_folder(f"{folder}/thermal_frames")

    thermal_tensor = []
    for frame_id, frame_bytes in enumerate(tqdm(Seq(seq_path))):
        frame_path = f"{folder}/fff_frames/{frame_id}.fff"
        with open(frame_path, "wb") as fff_file:
            fff_file.write(frame_bytes)

        thermal_image = get_thermal_image(frame_path)
        os.remove(frame_path)
        thermal_tensor.append(thermal_image)
        frame_path = f"{folder}/thermal_frames/{folder.split('/')[-1]}_{frame_id}.jpg"
        cv2.imwrite(frame_path, get_thermal_image_vis_gray(thermal_image))
    os.rmdir(f"{folder}/fff_frames")
    thermal_tensor = np.array(thermal_tensor)
    print(f"Saving thermal tensor {thermal_tensor.shape} ...")
    # np.save(f"{folder}/thermal.npy", thermal_tensor)
    np.savez_compressed(f"{folder}/thermal.npz", thermal_tensor)
    print("Complete!")


def make_empty_folder(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


if __name__ == "__main__":
    main()
