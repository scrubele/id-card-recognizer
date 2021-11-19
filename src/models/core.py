import os
import re

import difflib
import datefinder
import cv2
import numpy as np
from pytesseract import image_to_string
from fuzzysearch import find_near_matches
from fuzzywuzzy import process
# from src import TEST_FOLDER, logger

TEST_FOLDER = "data/test/"


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
    text_only = cv2.bitwise_and(img, img, mask=mask)


def preprocess_image(gray):
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)


    grad_x = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    cv2.imshow("sobel", grad_x)

    grad_x = np.absolute(grad_x)
    (minVal, maxVal) = (np.min(grad_x), np.max(grad_x))
    grad_x = (255 * ((grad_x - minVal) / (maxVal - minVal))).astype("uint8")
    cv2.imshow("grad_x", grad_x)

    grad_x = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, rect_kernel)

    cv2.imshow("thresh", grad_x)
    # cv2.waitKey(0)

    grad_x = cv2.GaussianBlur(grad_x, (7, 7), 0)
    binary = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # cv2.imshow("thresh", binary)
    # cv2.waitKey(0)

    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))

    dilation = cv2.dilate(binary, element1, iterations=1)
    # dilation = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, element1)
    erosion = cv2.GaussianBlur(dilation, (7, 7), 0)

    cv2.imshow("thresh", erosion)
    cv2.waitKey(0)

    dilation2 = cv2.dilate(erosion, element2, iterations=3)

    return erosion


def make_mask_image(gray_image):
    contours, hierarchy = cv2.findContours(
        gray_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )

    mask = np.zeros((gray_image.shape[0], gray_image.shape[1], 1), dtype=np.uint8)

    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        area = cv2.contourArea(contours[idx])
        cv2.drawContours(mask, [contours[idx]], -1, (255, 255, 255), -1)

    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    mask = cv2.dilate(mask, element2, iterations=1)
    cv2.imshow("mask", mask)
    cv2.waitKey(0)
    return mask


def recognise_text(mask, text_only):
    contours, hierarchy = cv2.findContours(
        mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )
    array_of_texts = []
    counter = 0
    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        # print(x, y, w, h )
        cropped_image = text_only[y : y + h, max(x - 10, 0) : x + w + 10]
        # print(cropped_image)
        text = image_to_string(cropped_image)
        text = text.split("\n")
        text = [
            element
            for element in text
            if element.strip() != "" and element.strip() != "\x0c"
        ]
        array_of_texts += text
        counter += 1
    return array_of_texts


def matches(large_string, query_string, threshold):
    words = large_string.split()
    for word in words:
        s = difflib.SequenceMatcher(None, word, query_string)
        match = "".join(word[i : i + n] for i, j, n in s.get_matching_blocks() if n)
        if len(match) / float(len(query_string)) >= threshold:
            yield match


def fuzzy_extract(ls, qs, threshold):
    for word, _ in process.extractBests(qs, (ls,), score_cutoff=threshold):
        # print('word {}'.format(word))
        for match in find_near_matches(qs, word, max_l_dist=1):
            match = word[match.start : match.end]
            # print('match {}'.format(match))
            index = ls.find(match)
            yield (match, index)


def clean_text(array_of_texts):
    array_of_text = " ".join(array_of_texts)
    text = re.sub("[^a-zA-Z0-9\n\.]", " ", array_of_text)
    text = text.lower()
    print(text)
    return text


def find_matches(large_string, query_strings):
    matches_count = []
    for query_string in query_strings:
        all_matches = list(matches(large_string, query_string, 0.8))
        match_count = len(all_matches)
        # matche = list(fuzzy_extract(query_string, large_string, 0))
        matches_count.append(match_count)
        print(all_matches)
    return matches_count


def detect_text_regions(image_path, query_strings=["javadi", "arjomand"]):
    image = cv2.imread(image_path)
    image = cv2.resize(image, None, fx=0.25, fy=0.25)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray_image = preprocess_image(gray)
    mask = make_mask_image(gray_image)
    text_only_image = cv2.bitwise_and(image, image, mask=mask)

    cv2.imshow("text_only_image", text_only_image)
    cv2.waitKey(0)

    array_of_texts = recognise_text(mask, text_only_image)
    text = clean_text(array_of_texts)
    print(array_of_texts)
    print(text)

    dates_format = list(datefinder.find_dates(text))
    dates = [date.strftime("%Y-%m-%d") for date in dates_format]
    print(dates)

    matches_count = find_matches(text, query_strings)
    matches_count = [1 for count in matches_count if count > 0]
    is_found = True if len(query_strings) == sum(matches_count) else False
    print(is_found)



if __name__ == "__main__":
    for i in range(18, 19):
        image_path = os.path.join(TEST_FOLDER, str(i) + ".jpg")
        print(image_path)
        # try:
        detect_text_regions(image_path)
        # except:
        #     pass
        cv2.destroyAllWindows()
        # cv2.waitKey(1)
