import cv2
import numpy as np

from src.data.preprocessors.detectors.contours import find_contours


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
    return x1, y1, x2, y2


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
