import os

import cv2
import fire

from src import TEST_FOLDER
from src.data.preprocessors.detectors.contours import process_contour_detection_approach, detect_linear_contours_approach
from src.data.preprocessors.detectors.hough_lines import process_hough_lines_approach


def preprocess_data(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, None, fx=0.5, fy=0.5)

    process_hough_lines_approach(img)
    detect_linear_contours_approach(img)
    process_contour_detection_approach(img)
    cv2.destroyAllWindows()


def test_image(image_name="1.jpg"):
    image_path = os.path.join(TEST_FOLDER, image_name)
    preprocess_data(image_path)


if __name__ == "__main__":
    fire.Fire(test_image)
