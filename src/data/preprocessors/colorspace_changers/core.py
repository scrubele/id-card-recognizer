import cv2
import numpy as np


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
