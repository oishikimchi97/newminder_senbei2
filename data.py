import argparse
from pathlib import Path

import cv2 as cv
import numpy as np


def crop_senbei(senbei_img: np.array, target_img: np.array) -> np.array:
    w, h = get_wh(senbei_img)
    top_left = match_template(senbei_img, target_img)
    print(top_left)
    cropped_img = target_img[top_left[1]:top_left[1] + h, top_left[0]: top_left[0] + w]
    return cropped_img


def get_wh(img: np.array):
    shape = img.shape
    if len(shape) == 2:
        w, h = img.shape[::-1]
    elif len(shape) == 3:
        _, w, h = img.shape[::-1]
    else:
        raise TypeError
    return w, h


def match_template(template_img: np.array, target_img: np.array):
    method = cv.TM_CCOEFF
    res = cv.matchTemplate(target_img, template_img, method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    return top_left


def draw_lines(senbei_img: np.array, x_poisition: int, width_list: list) -> list:
    _, w, h = senbei_img.shape[::-1]
    draw_position = ((x_poisition, 0), (x_poisition, h))

    line_images = list()

    for width in width_list:
        target = senbei_img.copy()
        cv.line(target, *draw_position, (255, 255, 255), width)
        line_images.append(target)

    return line_images

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--senbei_template",
        type=str,
        required=True,
        help=''
    )
    parser.add_argument(
        "--senbei_target",
        type=str,
        required=True,
        help=''
    )
    parser.add_argument(
        "--x_position",
        type=int,
        required=True,
        help=''
    )
    parser.add_argument(
        "--width_start",
        type=int,
        required=True,
        help=''
    )
    parser.add_argument(
        "--width_end",
        type=int,
        required=True,
        help=''
    )
    parser.add_argument(
        "--step_num",
        type=int,
        required=True,
        help=''
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='./senbei_output/',
        help=''
    )

    args = parser.parse_args()

    template_img = cv.imread(args.senbei_template)
    target_img = cv.imread(args.senbei_target)

    cropped_senbei = crop_senbei(template_img, target_img)

    width_list = list(range(args.width_start, args.width_end, args.step_num))

    line_images = draw_lines(cropped_senbei, args.x_position, width_list)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for img, line_width in zip(line_images, width_list):
        output_path = output_dir / f"drawed_senbei_line_{line_width}.bmp"

        cv.imwrite(str(output_path), img)
        print("Cropping and line Drawing completed!")
