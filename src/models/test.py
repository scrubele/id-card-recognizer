import os
import re

import cv2
import numpy as np
from pytesseract import image_to_string

TEST_FOLDER = "data/test/ids_/"


def detect_mser_regions(img):
    mser = cv2.MSER_create()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    regions, _ = mser.detectRegions(gray)

    vis = img.copy()
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

    cv2.polylines(vis, hulls, 1, (0, 255, 0))

    mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)

    for contour in hulls:
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
        # x, y, w, h = cv2.boundingRect(contour)
        # cropped_image = cv2.crop((x - 10, y, x + w + 10, y + h))
        # cropped_image = image[y:y + h, x-10:x + w+10]
        # str_store = re.sub(r'([^\s\w]|_)+', '', image_to_string(cropped_image))
        # print(str_store)
        # array_of_texts.append(str_store)
    # this is used to find only text regions, remaining are ignored
    text_only = cv2.bitwise_and(img, img, mask=mask)

    # cv2.imshow("text only", text_only)

    # cv2.waitKey(0)


def detect_text_regions(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, None, fx=0.25, fy=0.25)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)

    grad_x = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad_x = np.absolute(grad_x)

    grad_x = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, rect_kernel)
    grad_x = cv2.GaussianBlur(grad_x, (7, 7), 0)
    binary = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    cv2.imshow("thresh", binary)
    # cv2.waitKey(0)

    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))

    dilation = cv2.dilate(binary, element1, iterations=1)
    # dilation = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, element1)
    erosion = cv2.GaussianBlur(dilation, (7, 7), 0)

    dilation2 = cv2.dilate(erosion, element2, iterations=3)

    # cv2.imshow("dilation", dilation)
    cv2.imshow("erosion", erosion)
    # cv2.imshow("dilation2", dilation2)
    cv2.waitKey(0)

    contours, hierarchy = cv2.findContours(
        erosion.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )
    counter = 0

    mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)

    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        area = cv2.contourArea(contours[idx])

        # cropped_image = dilation.crop((x - 10, y, x + w + 10, y + h))
        cv2.drawContours(mask, [contours[idx]], -1, (255, 255, 255), -1)
        # cv2.drawContours(image, [contours[idx]], -1, [0, 255, 0], 2, 8)
        # cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)
        # cropped_image = image[y:y + h, x-10:x + w+10]
        # str_store = re.sub(r'([^\s\w]|_)+', '', image_to_string(cropped_image))
        # print(str_store)

    # image = cv2.resize(image, None, fx=0.25, fy=0.25)
    # mask = cv2.resize(mask, None, fx=0.25, fy=0.25)
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    mask = cv2.dilate(mask, element2, iterations=1)
    # cv2.imshow("dilation1", image)
    # cv2.imshow("mask", mask)
    text_only = cv2.bitwise_and(image, image, mask=mask)
    # cv2.imshow("text_only_rgb", text_only)
    # cv2.waitKey(0)
    img = text_only.copy()
    # detect_mser_regions(img)
    contours, hierarchy = cv2.findContours(
        mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )
    array_of_texts = []
    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        cropped_image = text_only[y : y + h, x - 10 : x + w + 10]
        text = image_to_string(cropped_image)
        text = text.split("\n")
        text = [
            element
            for element in text
            if element.strip() != "" and element.strip() != "\x0c"
        ]
        # str_store = re.sub(r'([^\s\w]|_)+', '', image_to_string(cropped_image))
        # print(text)
        array_of_texts += text
        counter += 1
    # print(array_of_texts)
    text = "".join(array_of_texts)
    print(text)
    # new_array = []
    # for text in array_of_texts:
    #     new_text = text.strip()
    #     if new_text != '' and new_text != '\x0c':
    #         new_array.append(new_text)
    # print(new_array)
    # if 'JAVADI' in text and  'ARJOMAND' in text:
    #     print('SUCCESS')


for i in range(18, 24):
    image_path = os.path.join(TEST_FOLDER, str(i) + ".png")
    print(image_path)
    # try:
    detect_text_regions(image_path)
    # except:
    #     pass
    cv2.destroyAllWindows()
    cv2.waitKey(1)
