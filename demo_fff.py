import cv2

from fe_tools.fff_tools import get_raw_image_np
from fe_tools.fff_tools import get_thermal_image
from fe_tools.fff_tools import get_thermal_image_vis


def main():
    fff_image = "examples/frame.fff"

    raw_image = get_raw_image_np(fff_image)
    print("Raw image:")
    print(raw_image)

    thermal_image = get_thermal_image(fff_image)
    print("Thermal image (Kelvin):")
    print(thermal_image)

    thermal_image = get_thermal_image(fff_image, is_celsius=False)
    print("Thermal image (Celsius):")
    print(thermal_image)

    thermal_image_vis = get_thermal_image_vis(thermal_image)

    # cv2.imwrite('thermal_img_vis.jpg', thermal_image_vis)
    cv2.imshow("Thermal image", thermal_image_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
