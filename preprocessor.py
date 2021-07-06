import cv2
import numpy as np
from skimage.transform import hough_line, hough_line_peaks, rotate
from skimage.feature import canny
from skimage.io import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from scipy.stats import mode
import imutils
from matplotlib import pyplot as plt
from scipy.ndimage import interpolation as inter
from PIL import Image as im
from copy import deepcopy

folder = "img/"
image_name = "test-1.jpg"


def scale_image(img: np.ndarray, scale_percent=40) -> np.ndarray:
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dimension = (width, height)
    img = cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)
    return img


def display_hough_lines(angles, rho):
    for angle, rho in zip(angles, dists):
        a = np.cos(angle)
        b = np.sin(angle)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow("image", img)
    cv2.waitKey(0)


def find_skew_angle(edges):
    tested_angles = np.deg2rad(np.arange(0.1, 180.0))
    hspace, theta, distances = hough_line(edges, theta=tested_angles)
    accum, angles, dists = hough_line_peaks(hspace, theta, distances)
    # display_hough_lines(angles, dists)
    most_common_angle = mode(np.around(angles, decimals=2))[0]
    skew_angle = np.rad2deg(most_common_angle - np.pi / 2)
    return skew_angle


def convert_to_edges(gray):
    kernel = np.ones((5, 5), np.float32) / 25
    gray = cv2.filter2D(gray, -1, kernel)
    edges = cv2.Canny(gray, 400, 600, apertureSize=5)
    cv2.imshow("image", edges)
    cv2.waitKey(0)
    return edges


def find_hough_lines(img, gray):
    """
    using OpenCV;
    """
    edges = convert_to_edges(gray)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 15)
    # print("lines-hough", lines)
    for i in range(8):
        for rho, theta in lines[i]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow("image", img)
    cv2.waitKey(0)


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def detect_contours(img, thresh_gray, is_skewed=True, to_display=True):
    input_img = deepcopy(img)
    contours, _ = cv2.findContours(thresh_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if to_display:
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # print(x, y, w, h)
            cv2.drawContours(input_img, [contour], 0, (0, 0, 255), 5)
            # cv2.imshow("input_img", input_img)
            # cv2.waitKey(0)
        cv2.imshow("Image", input_img)
        cv2.waitKey(0)
    max_contour_id = 2 if is_skewed else 1
    max_contour = contours[max_contour_id]
    rectangle = cv2.minAreaRect(max_contour)
    box_points = cv2.boxPoints(rectangle)
    return box_points


def crop_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh_gray = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    kernel = np.ones((3, 3), np.uint8)
    thresh_gray = cv2.morphologyEx(thresh_gray, cv2.MORPH_CLOSE, kernel)
    thresh_gray = cv2.morphologyEx(thresh_gray, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('Image_morph_gray.jpg', thresh_gray)
    # cv2.waitKey(0)

    box_points = detect_contours(img, thresh_gray)
    point_1, _, point_3, _ = box_points
    x1, x2 = sorted([int(point_1[0]), int(point_3[0])])
    y1, y2 = sorted([int(point_1[1]), int(point_3[1])])
    print(x1, x2, y1, y2)
    # Crop and save
    roi = img[y1:y2, x1:x2]
    cv2.imwrite(folder + "image-crop.jpg", roi)

    # Draw bounding box rectangle (debugging)
    cv2.rectangle(img, (x1, y1), (x2, y2), (200, 0, 0), 2)
    cv2.imwrite(folder + "image-cont.jpg", img)


if __name__ == "__main__":
    img = cv2.imread(folder + image_name)
    img = scale_image(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = canny(gray)
    best_angle = find_skew_angle(edges)[0]
    print(f"Best angle: {best_angle}")
    img = rotate_image(img, best_angle)
    cv2.imshow("skewed image", img)
    cv2.waitKey(0)

    crop_image(img)