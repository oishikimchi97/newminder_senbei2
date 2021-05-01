import argparse
from pathlib import Path

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def infer(args):
    target_img_path = args.target_img_path
    src = cv.imread(target_img_path)
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    gray = cv.bitwise_not(gray)
    bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                              cv.THRESH_BINARY, 15, -2)
    vertical = np.copy(bw)

    rows = vertical.shape[0]
    verticalsize = rows // 30

    verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalsize))

    vertical = cv.erode(vertical, verticalStructure)
    vertical = cv.dilate(vertical, verticalStructure)

    vertical = cv.bitwise_not(vertical)
    edges = cv.adaptiveThreshold(vertical, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                                 cv.THRESH_BINARY, 3, -2)

    kernel = np.ones((2, 2), np.uint8)
    edges = cv.dilate(edges, kernel)

    smooth = np.copy(vertical)

    smooth = cv.blur(smooth, (2, 2))

    (rows, cols) = np.where(edges != 0)
    vertical[rows, cols] = smooth[rows, cols]
    output_path = Path(args.output_dir) / Path(target_img_path).name
    cv.imwrite(str(output_path), vertical.astype(np.float32))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target_img_path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True
    )

    args = parser.parse_args()

    infer(args)
