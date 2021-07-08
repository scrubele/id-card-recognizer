import cv2
import numpy as np


def order_points(points):
    rectangle = np.zeros((4, 2), dtype="float32")
    s = points.sum(axis=1)
    diff = np.diff(points, axis=1)
    top_left = points[np.argmin(s)]
    top_right = points[np.argmin(diff)]
    bottom_right = points[np.argmax(s)]
    bottom_left = points[np.argmax(diff)]
    rectangle[0], rectangle[1], rectangle[2], rectangle[3] = top_left, top_right, bottom_right, bottom_left
    return rectangle


def four_point_transform(image, points, max_width, max_height):
    rect = order_points(points)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")
    transform_matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, transform_matrix, (max_width, max_height))
    return warped


def warp_image(image):
    input_image = image.copy()

    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (3, 3), 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    gray_img = cv2.morphologyEx(gray_img, cv2.MORPH_DILATE, kernel)
    gray_img = cv2.Canny(gray_img, 50, 150)
    cv2.imshow("gray", gray_img)
    contours, hierarchy = cv2.findContours(
        gray_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[0:6]
    for contour in contours:
        cv2.drawContours(image, [contour], -1, (255, 0, 0), 3)
        cv2.imshow("im", image)
        cv2.waitKey(0)

    print(len(contours))
    max_contour = contours[0]
    cv2.drawContours(image, [max_contour], -1, (0, 0, 255), 3)
    box_points = cv2.approxPolyDP(max_contour, 40, True).copy().reshape(-1, 2)
    # image = draw_points(image, box_points)
    warped = four_point_transform(image, box_points, input_image.shape[1], input_image.shape[0])

    # cv2.imshow("original", image)
    # cv2.imshow("warped", warped)
    # cv2.waitKey(0)
    return warped
