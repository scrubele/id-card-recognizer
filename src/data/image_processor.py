import os
from copy import deepcopy

import cv2
import fire
import numpy as np
from scipy.stats import mode
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks

from src import TEST_FOLDER


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


def find_hough_lines(gray):
    kernel = np.ones((5, 5), np.float32) / 25
    gray = cv2.filter2D(gray, -1, kernel)
    edges = cv2.Canny(gray, threshold1=200, threshold2=600, apertureSize=5)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=60)
    # lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=15, maxLineGap=50, minLineLength=50)
    return lines


def draw_hough_lines(input_img, lines, line_number=8):
    img = input_img.copy()
    for i in range(min(line_number, len(lines))):
        rho, theta = lines[i].ravel()
        x1, y1, x2, y2 = get_cartesian_coordinates(theta, rho)
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return img


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


def draw_intersections(input_img, Px, Py):
    img = input_img.copy()
    for cx, cy in zip(Px, Py):
        cx = np.round(cx).astype(int)
        cy = np.round(cy).astype(int)
        color = np.random.randint(0, 255, 3).tolist()  # random colors
        cv2.circle(
            img, (cx, cy), radius=4, color=color, thickness=-1
        )  # -1: filled circle
    return img


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


def draw_points(input_img, points_list, radius=4, color=[0, 0, 255], thickness=-1):
    img = input_img.copy()
    for points in points_list:
        cx, cy = points.ravel()
        cx = np.round(cx).astype(int)
        cy = np.round(cy).astype(int)
        cv2.circle(img, (cx, cy), radius=4, color=color, thickness=-1)
    return img


# POLYGONS


def get_black_picture(old):
    new = np.ones(old.shape, np.uint8)
    cv2.bitwise_not(new, new)
    return new


def filter_contours_by_size(contours):
    contours = filter(lambda cont: cv2.arcLength(cont, False) > 100, contours)
    contours = filter(lambda cont: cv2.contourArea(cont) > 10000, contours)
    return contours


def simplify_contours_down_to_polygons(contours):
    linear_contours = []
    for contour in contours:
        linear_contour = cv2.approxPolyDP(contour, 40, True).copy().reshape(-1, 2)
        linear_contours.append(linear_contour)
    return linear_contours


def draw_linear_contours(img, linear_contours):
    new = get_black_picture(img)
    cv2.drawContours(new, linear_contours, -1, (0, 255, 0), 1)
    cv2.GaussianBlur(new, (9, 9), 0, new)
    new = cv2.Canny(new, threshold1=0, threshold2=84, apertureSize=3)
    return new


def add_hough_lines_to_edges(edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 25)
    for line in lines[0]:
        x1, y1, x2, y2 = line
        cv2.line(edges, (x1, y1), (x2, y2), (255, 0, 0), 2, 8)
    return edges


def make_otsu_channel_binarization(img):
    morph = img.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gradient_image = cv2.morphologyEx(morph, cv2.MORPH_GRADIENT, kernel)
    image_channels = np.split(np.asarray(gradient_image), 3, axis=2)

    channel_height, channel_width, _ = image_channels[0].shape

    for i in range(0, 3):
        _, image_channels[i] = cv2.threshold(
            ~image_channels[i], 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY
        )
        image_channels[i] = np.reshape(
            image_channels[i], newshape=(channel_height, channel_width, 1)
        )

    image_channels = np.concatenate(
        (image_channels[0], image_channels[1], image_channels[2]), axis=2
    )
    return image_channels


# CONTOUR DETECTION


def erase_small_contours(input_img, contours, max_area=10, max_aspect_ratio=1.5):
    """
    Erasing contours which have ratio closed to square
    """
    img = input_img.copy()
    for contour in contours:
        area = cv2.contourArea(contour)
        (x, y), (w, h), angle = cv2.minAreaRect(contour)
        aspect_ratio = max(w, h) / min(w, h)
        if area < max_area or aspect_ratio < max_aspect_ratio:
            convex = cv2.convexHull(contour, False)
            cv2.fillPoly(img, pts=[contour], color=0)
            cv2.fillConvexPoly(img, np.array(convex, "int32"), color=0)
    return img


def draw_contours(input_img, contours):
    img = input_img.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.drawContours(img, [contour], 0, (0, 0, 255), 5)
    return img


def find_contours(input_img, gray_image, to_draw=True):
    gray_img = gray_image.copy()
    contours, hierarchy = cv2.findContours(
        gray_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[1:]
    if to_draw:
        contour_image = draw_contours(input_img, contours)
    return contours, contour_image


def adjust_contours(gray_image, contours):
    gray_img = gray_image.copy()
    contour_image = erase_small_contours(gray_img, contours)
    contour_image = cv2.morphologyEx(
        contour_image,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    # cv2.imshow('input_contour_image', gray_img)
    # cv2.imwrite(DEBUG_FOLDER + image_name + "_" + "gray_img.png", gray_img)
    return contour_image


def find_closed_contours(input_img, contours):
    img = input_img.copy()
    hull = []

    for i in range(len(contours)):
        hull.append(cv2.convexHull(contours[i], False))

    for i in range(len(contours)):
        cv2.drawContours(img, contours, i, [0, 255, 0], 2, 8)
        cv2.drawContours(img, hull, i, [255, 0, 0], 2, 8)
    return img


def apply_morphology(input_gray):
    gray = input_gray.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gray = cv2.morphologyEx(gray, cv2.MORPH_ERODE, kernel)
    # cv2.imwrite(DEBUG_FOLDER + image_name + "_" + "morph.png", gray)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    # cv2.imwrite(DEBUG_FOLDER + image_name + "_" + "morph_1.png", gray)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    # cv2.imwrite(DEBUG_FOLDER + image_name + "_" + "morph_2.png", gray)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gray = cv2.morphologyEx(gray, cv2.MORPH_ERODE, kernel)
    # cv2.imwrite(DEBUG_FOLDER + image_name + "_" + "morph_3.png", gray)
    return gray


# ROTATE IMAGE


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


def detect_crop_box_points(img, gray_image, is_skewed=True, to_display=True):
    input_img = img.copy()
    contours, contour_image = find_contours(input_img, gray_image, to_display)
    max_contour_id = 1 if is_skewed else 0  # skip the first contour
    max_contour = contours[max_contour_id]
    rectangle = cv2.minAreaRect(max_contour)
    box_points = cv2.boxPoints(rectangle)
    return box_points


def crop_by_coordinates(input_img, coordinates):
    img = input_img.copy()
    x1, y1, x2, y2 = coordinates
    roi = img[y1:y2, x1:x2]

    cv2.rectangle(img, (x1, y1), (x2, y2), (200, 0, 0), 2)
    # cv2.imshow("cropped_image", roi)
    # cv2.waitKey(0)
    # cv2.imwrite(DEBUG_FOLDER + "image-cont.jpg", img)
    return roi


def crop_image(img, image_name=""):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh_gray = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    kernel = np.ones((3, 3), np.uint8)
    thresh_gray = cv2.morphologyEx(thresh_gray, cv2.MORPH_CLOSE, kernel)
    thresh_gray = cv2.morphologyEx(thresh_gray, cv2.MORPH_OPEN, kernel)

    box_points = detect_crop_box_points(img, thresh_gray)
    point_1, _, point_3, _ = box_points
    x1, x2 = sorted([int(point_1[0]), int(point_3[0])])
    y1, y2 = sorted([int(point_1[1]), int(point_3[1])])
    x1, y1, x2, y2 = crop_by_coordinates(img, [x1, y1, x2, y2])


def clip_by_coordinates(input_img, polygon_coord, name=""):
    img = input_img.copy()

    x, y, w, h = cv2.boundingRect(polygon_coord)
    box_coordinates = [x, y, x + w, y + h]
    cropped_coord = crop_by_coordinates(img, box_coordinates)

    polygon_coord = polygon_coord - polygon_coord.min(axis=0)
    mask = np.zeros(cropped_coord.shape[:2], np.uint8)
    cv2.drawContours(mask, [polygon_coord.astype(int)], -1, (255, 255, 255), -1, cv2.LINE_AA)

    black_clipped = cv2.bitwise_and(cropped_coord, cropped_coord, mask=mask)
    white_background = np.ones_like(cropped_coord, np.uint8) * 255
    cv2.bitwise_not(white_background, white_background, mask=mask)
    white_clipped = white_background + black_clipped
    return white_clipped


# APPROACHES


def process_hough_lines_approach(img, image_name=""):
    input_img = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 3)
    edges = cv2.Canny(gray, 50, 150)
    dilated = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8))

    hough_lines = find_hough_lines(gray)
    # hough_line_img = draw_hough_lines(input_img, hough_lines, line_number=8)
    # cv2.imshow("Hough lines", hough_line_img)
    # cv2.imwrite(DEBUG_FOLDER + image_name + "_dilated.png", img)

    horizontal_lines, vertical_lines = segment_lines_by_direction_by_polar_coord(
        hough_lines, delta=800
    )
    # segmented_lines = draw_segmented_lines(input_img, horizontal_lines, vertical_lines)
    # cv2.imshow("Segmented Hough Lines", segmented_lines)
    # cv2.imwrite(DEBUG_FOLDER + image_name + "_" + "hough.png", segmented_lines)

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
    # cv2.imshow("Points", points_image)
    # cv2.waitKey(0)
    # cv2.imwrite(DEBUG_FOLDER + image_name + "_" + "points.png", points_image)

    skew_angle = find_skew_angle(img, edges)
    # print(f"Best angle:{skew_angle}")


def detect_linear_contrours_approach(input_img, image_name=""):
    img = input_img.copy()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = gray_img.copy()

    img = cv2.GaussianBlur(img, (3, 3), 0, img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(9, 9))
    dilated = cv2.dilate(img, kernel)
    edges = cv2.Canny(dilated, threshold1=0, threshold2=84, apertureSize=3)
    # cv2.imshow("edges", edges)
    # cv2.waitKey(0)

    edges = add_hough_lines_to_edges(edges)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    linear_contours = simplify_contours_down_to_polygons(contours)
    linear_contours_image = draw_linear_contours(img, linear_contours)
    # cv2.imshow("linear_contours_image", linear_contours_image)
    # cv2.waitKey(0)


def process_threshold_approach(img, image_name="1.jpg"):
    input_img = img.copy()

    binarized = make_otsu_channel_binarization(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binarized_gray = cv2.morphologyEx(binarized, cv2.MORPH_ERODE, kernel)
    binarized_gray = cv2.cvtColor(binarized_gray, cv2.COLOR_BGR2GRAY)
    binarized_gray = cv2.dilate(binarized_gray, kernel)
    # cv2.imwrite(DEBUG_FOLDER + image_name + "_" + "otsu_binarized.png", binarized)
    # cv2.imwrite(DEBUG_FOLDER + image_name + "_" + "otsu_binarized_gray.png", binarized_gray)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5
    )
    # cv2.imwrite(DEBUG_FOLDER + image_name + "_" + "adaptiveThreshold.png", gray)

    # gray = cv2.GaussianBlur(gray, (5, 5), 7)
    # edges = cv2.Canny(gray, 50, 150)
    # dilated = cv2.dilate(gray, np.ones((3, 3), dtype=np.uint8))

    # gray = apply_morphology(binarized_gray)
    gray = apply_morphology(gray)

    contours, contour_image = find_contours(input_img, binarized_gray)
    contour_image = adjust_contours(binarized_gray, contours)
    # cv2.imshow("contour_image", contour_image)
    # cv2.imwrite(DEBUG_FOLDER + image_name + "_" + "contours.png", contour_image)

    closed_contour_image = find_closed_contours(input_img, contours)
    # cv2.imshow("convex", closed_contour_image)
    # cv2.waitKey(0)


def process_hsv_approach(input_img, image_name=""):
    img = input_img.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    ret, thresh_H = cv2.threshold(hsv[:, :, 0], 0, 255, cv2.THRESH_OTSU)
    ret, thresh_S = cv2.threshold(
        hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(thresh_H + thresh_S, kernel, iterations=1)

    # cv2.imshow("hsv", hsv)
    # cv2.imshow("H", hsv[:, :, 0])
    # cv2.imshow("S", hsv[:, :, 1])
    # cv2.imshow("thresh", thresh_H + thresh_S)
    # cv2.imshow("dilation", thresh)
    # cv2.waitKey(0)
    # cv2.imwrite(DEBUG_FOLDER + image_name + "_" + "dilated.png", thresh)


def process_cropping(input_img, image_name=""):
    img = input_img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = canny(gray)
    best_angle = find_skew_angle(img, edges)[0]
    # print(f"Best angle: {best_angle}")
    img = rotate_image(img, best_angle)
    crop_image(img, image_name)
    # cv2.imshow("skewed image", img)
    # cv2.waitKey(0)


def preprocess_data(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, None, fx=0.5, fy=0.5)

    process_hough_lines_approach(img)
    detect_linear_contrours_approach(img)
    process_threshold_approach(img)
    process_hsv_approach(img)
    # process_cropping(img) # incorrect regions detection so far
    cv2.destroyAllWindows()


def test_image(image_name="1.jpg"):
    image_path = os.path.join(TEST_FOLDER, image_name)
    preprocess_data(image_path)


if __name__ == "__main__":
    fire.Fire(test_image)
