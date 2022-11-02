import os
import cv2
import shutil
import numpy as np

from tqdm import tqdm

from fe_tools.seq import Seq
from fe_tools.fff_tools import get_raw_image_np
from fe_tools.fff_tools import get_thermal_image
from fe_tools.fff_tools import get_thermal_image_vis


def main():
    seq_path = "examples/SEQ_0102.seq"
    folder = seq_path.split(".")[0]
    make_empty_folder(folder)
    make_empty_folder(f"{folder}/fff_frames")
    make_empty_folder(f"{folder}/thermal_frames")

    result_tensor_th = []
    result_tensor_raw = []
    for frame_id, frame_bytes in enumerate(tqdm(Seq(seq_path))):
        frame_path = f"{folder}/fff_frames/{frame_id}.fff"
        with open(frame_path, "wb") as fff_file:
            fff_file.write(frame_bytes)

        raw_image = get_raw_image_np(frame_path)
        thermal_image = get_thermal_image(frame_path)

        result_tensor_th.append(thermal_image)
        result_tensor_raw.append(raw_image)

        cv2.imwrite(f"{folder}/thermal_frames/{frame_id}.jpg", get_thermal_image_vis(thermal_image))

    result_tensor_th = np.array(result_tensor_th)
    result_tensor_raw = np.array(result_tensor_raw)

    print(f"Shape of thermal tensor: {result_tensor_th.shape}")
    print(f"Shape of raw tensor: {result_tensor_raw.shape}")

    np.savetxt(f"{folder}/thermal.txt", result_tensor_th.reshape(-1), fmt='%d')
    np.savetxt(f"{folder}/raw.txt", result_tensor_raw.reshape(-1), fmt='%d')


def make_empty_folder(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


# def save_np_arrays_to_txt(path: str, result_tensor_th: np.ndarray, result_tensor_raw: np.ndarray) -> None:
#     save_np_array(f"{path}/thermal.txt", result_tensor_th)
#     save_np_array(f"{path}/raw.txt", result_tensor_raw)
#
#
# def save_np_array(path: str, array: np.ndarray) -> None:
#     with open(path, "w") as f:
#         for frame_id, frame in enumerate(array):
#             f.write(f"Frame {frame_id}: \n")
#             for row in frame:
#                 for pixel in row:
#                     f.write(f"{pixel} ")
#                 f.write("\n")


if __name__ == "__main__":
    main()
