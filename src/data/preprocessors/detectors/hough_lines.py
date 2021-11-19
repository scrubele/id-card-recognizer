from copy import deepcopy

import cv2
import numpy as np
from scipy.stats import mode
from skimage.transform import hough_line, hough_line_peaks

from src import logger


def get_cartesian_coordinates(angle, rho):
    a = np.cos(angle)
    b = np.sin(angle)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    return x1, y1, x2, y2


def find_skew_angle(img, edges):
    tested_angles = np.deg2rad(np.arange(0.1, 180.0))
    hspace, theta, distances = hough_line(edges, theta=tested_angles)
    accum, angles, dists = hough_line_peaks(hspace, theta, distances)
    most_common_angle = mode(np.around(angles, decimals=2))[0]
    skew_angle = np.rad2deg(most_common_angle - np.pi / 2)
    lines = np.float32(np.column_stack((dists, angles)))
    hough_lines_img = draw_hough_lines(img, lines)
    # cv2.imshow("hough_lines_img", hough_lines_img)
    # cv2.waitKey(0)
    return skew_angle


def find_hough_lines(gray):
    kernel = np.ones((5, 5), np.float32) / 25
    gray = cv2.filter2D(gray, -1, kernel)
    edges = cv2.Canny(gray, threshold1=200, threshold2=600, apertureSize=5)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=60)
    # lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=15, maxLineGap=50, minLineLength=50)
    return lines


def segment_lines_by_direction_by_polar_coord(hough_lines, delta, max_line_number=20):
    lines = deepcopy(hough_lines)[:max_line_number]
    horizontal_lines = []
    vertical_lines = []
    for line_id, line in enumerate(lines):
        rho, angle = line[0]
        x1, y1, x2, y2 = get_cartesian_coordinates(angle, rho)
        if abs(x2 - x1) < delta:
            vertical_lines.append([x1, y1, x2, y2])
        elif abs(y2 - y1) < delta:
            horizontal_lines.append([x1, y1, x2, y2])
    return horizontal_lines, vertical_lines


def find_lines_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    Px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
            (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    )
    Py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
            (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    )
    return Px, Py


def find_all_lines_intersection(horizontal_lines, vertical_lines):
    Px = []
    Py = []
    for h_line in horizontal_lines:
        for v_line in vertical_lines:
            px, py = find_lines_intersection(h_line, v_line)
            Px.append(px)
            Py.append(py)
    intersection_points = np.float32(np.column_stack((Px, Py)))
    return intersection_points


def find_cluster_points(points, nclusters=4):
    try:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, _, centers = cv2.kmeans(
            points, nclusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )
    except:
        centers = points
    return centers


def find_extreme_points(P):
    try:
        top_right = P.sum(axis=1).argmax()
        bottom_left = P.sum(axis=1).argmin()
        top_left = (P[:, 0] + P.max(axis=0)[1] - P[:, 1]).argmin()
        bottom_right = (P.max(axis=0)[0] - P[:, 0] + P[:, 1]).argmax()
        extreme_points = [P[top_right], P[bottom_left], P[top_left], P[bottom_right]]
    except:
        extreme_points = []
    return extreme_points


def draw_hough_lines(input_img, lines, line_number=8):
    img = input_img.copy()
    for i in range(min(line_number, len(lines))):
        rho, theta = lines[i].ravel()
        x1, y1, x2, y2 = get_cartesian_coordinates(theta, rho)
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return img


def draw_segmented_lines(input_img, horizontal_lines, vertical_lines):
    img = input_img.copy()
    for line in horizontal_lines:
        x1, y1, x2, y2 = line
        color = [0, 0, 255]
        cv2.line(img, (x1, y1), (x2, y2), color=color, thickness=1)
    for line in vertical_lines:
        x1, y1, x2, y2 = line
        color = [255, 0, 0]
        cv2.line(img, (x1, y1), (x2, y2), color=color, thickness=1)
    return img


def draw_points(input_img, points_list, radius=4, color=[0, 0, 255], thickness=-1):
    img = input_img.copy()
    for points in points_list:
        cx, cy = points.ravel()
        cx = np.round(cx).astype(int)
        cy = np.round(cy).astype(int)
        cv2.circle(img, (cx, cy), radius=4, color=color, thickness=-1)
    return img


def process_hough_lines_approach(img, image_name=""):
    input_img = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 3)
    edges = cv2.Canny(gray, 50, 150)
    dilated = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8))

    hough_lines = find_hough_lines(gray)
    hough_line_img = draw_hough_lines(input_img, hough_lines, line_number=8)
    logger.log_image(image_name, stage_name="hough_lines", image=hough_line_img, save=False)

    horizontal_lines, vertical_lines = segment_lines_by_direction_by_polar_coord(
        hough_lines, delta=800
    )
    segmented_lines = draw_segmented_lines(input_img, horizontal_lines, vertical_lines)
    logger.log_image(image_name, stage_name="segmented_lines", image=segmented_lines, save=False)

    intersection_points = find_all_lines_intersection(horizontal_lines, vertical_lines)
    center_points = find_cluster_points(intersection_points, nclusters=4)
    extreme_points = find_extreme_points(intersection_points)
    corner_points = cv2.goodFeaturesToTrack(edges, 4, 0.5, 50)

    points_image = input_img.copy()
    points_image = draw_points(
        points_image, intersection_points, radius=2, color=[128, 128, 128]
    )
    points_image = draw_points(points_image, center_points, color=[255, 0, 0])
    points_image = draw_points(points_image, extreme_points, color=[0, 255, 0])
    points_image = draw_points(points_image, corner_points, color=[0, 0, 255])
    logger.log_image(image_name, stage_name="points", image=points_image, save=True)

    skew_angle = find_skew_angle(img, edges)
    # print(f"Best angle:{skew_angle}")
