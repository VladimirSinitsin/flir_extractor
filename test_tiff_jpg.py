import cv2
import numpy as np
import tifffile as tiff

from glob import glob

DIFF_THRESHOLD = 10

# TODO: DIFF_THRESHOLD doesn't work properly

def main():
    jpg_files = sorted(glob("erg_kz/1_day/SEQ_0936/*.jpg"))
    tiff_files = sorted(glob("erg_kz/1_day/SEQ_0936/*.tiff"))
    for jpg_path, tiff_path in zip(jpg_files, tiff_files):
        print(f"Comparing {jpg_path.split('/')[-1]} and {tiff_path.split('/')[-1]}:")
        jpg_img = cv2.imread(jpg_path, cv2.IMREAD_GRAYSCALE)
        tiff_img = tiff.imread(tiff_path)
        if not np.all(np.abs(jpg_img - tiff_img) > DIFF_THRESHOLD):
            print(f"Shape of jpg_img: {jpg_img.shape}")
            print(f"Shape of tiff_img: {tiff_img.shape}")
            diff_img = cv2.cvtColor(jpg_img, cv2.COLOR_GRAY2BGR)
            diff_img[np.where(np.abs(jpg_img - tiff_img) > DIFF_THRESHOLD)] = np.array([0, 0, 255])
            images = np.concatenate(
                    (
                        cv2.cvtColor(jpg_img, cv2.COLOR_GRAY2BGR),
                        cv2.cvtColor(tiff_img, cv2.COLOR_GRAY2BGR),
                        diff_img
                    ),
                    axis=1
            )
            cv2.imshow("Images", images)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == ord("q"):
                break


if __name__ == "__main__":
    main()
