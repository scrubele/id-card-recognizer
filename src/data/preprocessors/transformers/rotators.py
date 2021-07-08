import cv2
import numpy as np

from src.data.preprocessors.detectors.contours import find_contours
from src.data.preprocessors.detectors.hough_lines import find_skew_angle


def rotate_image_by_angle(image, angle, expand=True):
    height, width = image.shape[:2]
    image_center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(image_center, angle, 1.)
    if not expand:
        bound_w, bound_h = image.shape[1::-1]
    else:
        abs_cos = abs(rotation_matrix[0, 0])
        abs_sin = abs(rotation_matrix[0, 1])
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)
        rotation_matrix[0, 2] += bound_w / 2 - image_center[0]
        rotation_matrix[1, 2] += bound_h / 2 - image_center[1]
    result_image = cv2.warpAffine(image, rotation_matrix, (bound_w, bound_h), flags=cv2.INTER_LINEAR)
    return result_image


def rotate_image(image, angle=0, white_background=True):
    if angle == 0:
        edges = cv2.Canny(image, threshold1=0, threshold2=84, apertureSize=3)
        angle = int(find_skew_angle(image, edges))
    result = rotate_image_by_angle(image, angle, expand=True)
    if white_background:
        mask = np.zeros(result.shape[:2], np.uint8)
        white_background = np.ones_like(result, np.uint8) * 255
        cv2.bitwise_not(white_background, white_background, mask=mask)
        result = white_background + result
    return result, angle
