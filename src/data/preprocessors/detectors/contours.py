import cv2
import numpy as np


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


def process_contour_detection_approach(img, image_name="1.jpg"):
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


def detect_linear_contours_approach(input_img, image_name=""):
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
