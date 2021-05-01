import argparse
import glob
import math

import cv2
import numpy as np
import cv2 as cv
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
from matplotlib import pyplot as plt

import data


def expand_img(img: np.array, frame_w, frame_h):
    frame = np.zeros((frame_w, frame_h))
    img_w, img_h = data.get_wh(img)
    frame[:img_h, :img_w] = img
    return frame


class ImgFrame:
    def __init__(self, img: np.array, frame_shape: tuple):
        """

        :type frame_shape: (w, h)

        """
        self.gray = True if len(img.shape) == 2 else False
        if self.gray:
            self.img_frame = np.zeros(*frame_shape)
        else:
            self.img_frame = np.zeros((*frame_shape, 3))

        self._img = img
        self.w, self.h = data.get_wh(img)
        self.frame_shape = frame_shape
        self.img_center = (int(frame_shape[0] / 2), int(frame_shape[1] / 2))
        self.img_top_left = (int((frame_shape[0] - self.w) / 2), int((frame_shape[1] - self.h) / 2))
        self.img_frame_init()

    @property
    def img(self, ):
        return self._img

    @img.setter
    def img(self, img):
        self.w, self.h = data.get_wh(img)
        self._img = img
        self.img_top_left = (self.img_center[0] - int(self.w / 2), self.img_center[1] - int(self.h / 2))
        self.img_frame_init()

    def img_frame_init(self):
        self.img_frame[:] = 0
        if self.gray:
            self.img_frame[self.img_top_left[1]:self.h + self.img_top_left[1],
            self.img_top_left[0]:self.w + self.img_top_left[0]] = self._img
        else:
            self.img_frame[self.img_top_left[1]:self.h + self.img_top_left[1],
            self.img_top_left[0]:self.w + self.img_top_left[0],
            :] = self._img

    def move_img(self, x=0, y=0):  # TODO: Add the exception for movement getting out the frame.
        self.img_frame[:] = 0
        x, y = int(x), int(y)
        self.img_center = (self.img_center[0] + x, self.img_center[1] + y)
        self.img_top_left = (self.img_top_left[0] + x, self.img_top_left[1] + y)
        if self.gray:
            self.img_frame[self.img_top_left[1]: self.img_top_left[1] + self.h,
            self.img_top_left[0]: self.img_top_left[0] + self.w] = self.img
        else:
            self.img_frame[self.img_top_left[1]: self.img_top_left[1] + self.h,
            self.img_top_left[0]: self.img_top_left[0] + self.w, :] = self.img

    def rotate_image(self, angle: int): # TODO: update top_left by rotation.
        rot_mat = cv2.getRotationMatrix2D(self.img_center, angle, 1.0)
        self.img_frame = cv2.warpAffine(self.img_frame, rot_mat, self.img_frame.shape[1::-1])


def get_offset(template_imgs: np.array, base_img: np.array, target_img: np.array) -> list:
    """

    :rtype: offset: base top left point - target_top_left
    """
    base_top_lefts = []
    target_top_lefts = []
    for template_img in template_imgs:
        base_top_left = data.match_template(template_img, base_img)
        target_top_left = data.match_template(template_img, target_img)
        print("base_top_left:", base_top_left)
        print("target_top_left:", target_top_left)
        base_top_lefts.append(base_top_left)
        target_top_lefts.append(target_top_left)
    base_top_left = np.array(base_top_lefts).mean(axis=0)
    target_top_left = np.array(target_top_lefts).mean(axis=0)

    offset = [target_top_left[i] - base_top_left[i] for i in range(len(base_top_left))]
    return offset

def substract(args):
    template_paths = glob.glob(args.template_dir + '/*')
    template_imgs = []
    for path in template_paths:
        template_img = cv.imread(path)
        template_imgs.append(template_img)

    base_img = cv.imread(args.base_img_path)
    target_img = cv.imread(args.target_img_path)

    base_w, base_h = data.get_wh(base_img)
    target_w, target_h = data.get_wh(target_img)

    max_w, max_h = max(base_w, target_w), max(base_h, target_h)
    base_img_frame = ImgFrame(base_img, (2 * max_w, 2 * max_h))
    target_img_frame = ImgFrame(target_img, (2 * max_w, 2 * max_h))

    offsets = get_offset(template_imgs, base_img_frame.img, target_img_frame.img)
    print(f"offset: {offsets}")

    base_move = [offset if offset > 0 else 0 for offset in offsets]
    target_move = [-offset if offset <= 0 else 0 for offset in offsets]

    base_img_frame.move_img(*base_move)
    target_img_frame.move_img(*target_move)

    # Fit two images by rotation
    lowest_error = math.inf
    best_angle = 0
    max_angle = 30
    step = 2
    # TODO: change to Parallel Processing.
    for angle in range(-max_angle, max_angle + 1, step):
        if angle == -max_angle:
            base_img_frame.rotate_image(-max_angle)
        else:
            base_img_frame.rotate_image(step)
        error = mse(base_img_frame.img_frame, target_img_frame.img_frame)
        if error < lowest_error:
            lowest_error = error
            best_angle = angle
    print(f"lowest:error is {lowest_error} in {best_angle} degree.")

    # return to best angle
    base_img_frame.rotate_image(best_angle - max_angle)

    threshold = 50
    sub_img_frame = (np.abs(target_img_frame.img_frame - base_img_frame.img_frame) > threshold) * 255

    # inter_mask_frame = base_img_frame.img_frame * target_img_frame.img_frame > 0
    # sub_img_frame = sub_img_frame * inter_mask_frame

    # cv.imwrite("base_img.bmp", base_img_frame.img_frame.astype(np.float32))
    # cv.imwrite("target_img.bmp", target_img_frame.img_frame.astype(np.float32))
    cv.imwrite(args.sub_path, sub_img_frame.astype(np.float32))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--template_dir",
        type=str,
        required=True,
        help=''
    )
    parser.add_argument(
        "--base_img_path",
        type=str,
        required=True,
        help=''
    )
    parser.add_argument(
        "--target_img_path",
        type=str,
        required=True,
        help=''
    )
    parser.add_argument(
        "--sub_path",
        type=str,
        required=True,
        help=''
    )

    args = parser.parse_args()
    substract(args)
