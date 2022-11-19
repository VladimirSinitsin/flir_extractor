import io
import os
import re
import sys
import json
import subprocess
import numpy as np

from math import exp
from math import log
from math import inf

from PIL import Image


EXIFTOOL_EXISTS = False


def get_thermal_image(fff_img_filename: str, is_celsius: bool = True) -> np.ndarray:
    """
    Get the temperature image from the fff image
    :param fff_img_filename: path to image
    :param is_celsius: if the temperature on the image in celsius
    :return:
    """
    if not EXIFTOOL_EXISTS:
        check_exiftool()
    meta = get_meta(fff_img_filename)
    raw_image = get_raw_image_np(fff_img_filename)
    thermal_np = np.vectorize(_raw2temperature)(
            raw_image,
            meta['PlanckR1'],
            meta['PlanckR2'],
            meta['PlanckB'],
            meta['PlanckO'],
            meta['PlanckF'],
            meta['Emissivity'],
            _extract_float(meta['ReflectedApparentTemperature']),
    )
    # Convert to Celsius
    thermal_np = thermal_np - np.min(thermal_np) if not is_celsius else thermal_np

    return thermal_np


def get_thermal_image_vis(thermal_np: np.ndarray) -> np.ndarray:
    """ Get the visualized temperature image from the numpy array of temperatures """
    thermal_img_vis = np.zeros((thermal_np.shape[0], thermal_np.shape[1], 3), dtype=np.uint8)
    thermal_np -= np.min(thermal_np)
    thermal_np = (thermal_np / np.max(thermal_np) * 255).astype("uint8")
    thermal_img_vis[:, :, 2] = thermal_np
    return thermal_img_vis


def get_thermal_image_vis_gray(thermal_np: np.ndarray, min_thr: float, max_thr: float) -> np.ndarray:
    th_np = thermal_np.copy()
    # th_np -= np.min(th_np)
    # th_np -= 200
    th_np = np.clip(th_np, min_thr, max_thr)
    thermal_img_vis = (th_np / np.max(th_np) * 255).astype("uint8")
    return thermal_img_vis


def get_meta(fff_img_filename: str) -> dict:
    """ Get the metadata from the fff image """
    meta_json = subprocess.check_output(
            ['exiftool', fff_img_filename, '-Emissivity', '-SubjectDistance', '-AtmosphericTemperature',
             '-ReflectedApparentTemperature', '-IRWindowTemperature', '-IRWindowTransmission', '-RelativeHumidity',
             '-PlanckR1', '-PlanckB', '-PlanckF', '-PlanckO', '-PlanckR2', '-j']
    )
    meta = json.loads(meta_json.decode())[0]
    return meta


def get_raw_image_np(fff_img_filename: str) -> np.ndarray:
    """ Get the raw image from the fff image """
    thermal_img_bytes = subprocess.check_output(['exiftool', "-RawThermalImage", "-b", fff_img_filename])
    thermal_img_stream = io.BytesIO(thermal_img_bytes)

    thermal_img = Image.open(thermal_img_stream)
    thermal_np = np.array(thermal_img)
    return thermal_np


def _extract_float(dirty_str: str) -> float:
    """ Extract the float value of a string, helpful for parsing the exiftool data """
    digits = re.findall(r"[-+]?\d*\.\d+|\d+", dirty_str)
    return float(digits[0])


def _raw2temperature(raw, pr1, pr2, pb, po, pf, e, r_temp):
    """
    Convert single raw value to temperature
    https://exiftool.org/forum/index.php?msg=23944
    """
    raw_refl = pr1 / (pr2 * (exp(pb / (r_temp + 273.15)) - pf)) - po
    raw_obj = (raw - (1 - e) * raw_refl) / e
    ln_arg = pr1 / (pr2 * (raw_obj + po)) + pf
    degree = pb / log(ln_arg) - 273.15 if ln_arg > 0 else pb / -inf - 273.15
    return degree


def check_exiftool():
    """ Check if exiftool is installed """
    global EXIFTOOL_EXISTS

    # check os
    if os.name == "posix":
        try:
            subprocess.check_output(["exiftool", "-ver"])
            print("OK. exiftool found")
            EXIFTOOL_EXISTS = True
        except FileNotFoundError:
            print("To work you should install exiftool via apt-get")
            print("sudo apt install libimage-exiftool-perl")
    elif os.name == "nt":
        if os.path.isfile("bin/exiftool.exe"):
            print('OK. exiftool.exe found in ./bin dir')
            EXIFTOOL_EXISTS = True
        else:
            print(
                "To work you should Download exiftool.exe binary "
                "from here https://exiftool.org/ and put it in ./bin dir. "
                "Additional instruction: https://zalinux.ru/?p=5033"
            )
            sys.exit(1)
    else:
        print('Other OS are not supported')
        print('Runs on windows or Linux')
        sys.exit(1)
