import os

import cv2
import fire

from src import TEST_FOLDER, DEBUG_FOLDER, TEMPLATES_FOLDER
from src.data.preprocessors.detectors.hough_lines import find_skew_angle
from src.data.preprocessors.matchers.feature.flann import flann_matcher
from src.data.preprocessors.transformers.croppers import clip_by_coordinates
from src.data.preprocessors.transformers.rotators import rotate_image


def is_card_detected(polygon_corners):
    if len(polygon_corners) > 0:
        negative_values = polygon_corners.ravel()
        negative_values = negative_values[negative_values < 0]
        if len(negative_values) > 0:
            return False
        else:
            return True
    else:
        return False


def process_image(template_image, input_image, image_name, template_name):
    polygon_corners = flann_matcher(template_image, input_image, image_name=image_name)
    if is_card_detected(polygon_corners):
        clipped_image = clip_by_coordinates(input_image, polygon_corners, name="")
        # cv2.imshow(image_name + ".clipped_image", clipped_image)
        # cv2.imwrite(os.path.join(DEBUG_FOLDER, image_name + "_clipped.png"), clipped_image)
        # cv2.waitKey(0)

        rotated_image, angle = rotate_image(clipped_image)
        print(f"Angle: {angle}")
        # cv2.imshow(image_name + ".rotated_image", rotated_image)
        cv2.imwrite(os.path.join(DEBUG_FOLDER, image_name + "_rotated_image.png"), rotated_image)
        # cv2.waitKey(0)

    else:
        print("Id card is not detected")


def preprocess_data(image_path, template_path=str(os.path.join(TEMPLATES_FOLDER, "template.jpg"))):
    image_name = os.path.basename(image_path)
    template_name = os.path.basename(image_path)

    template_image = cv2.imread(template_path)
    input_image = cv2.imread(image_path)

    template_image = cv2.resize(template_image, None, fx=0.5, fy=0.5)
    input_image = cv2.resize(input_image, None, fx=0.5, fy=0.5)

    process_image(template_image, input_image, image_name, template_name)
    cv2.destroyAllWindows()


def test_image(image_name="1.jpg"):
    image_path = str(os.path.join(TEST_FOLDER, image_name))
    preprocess_data(image_path)


if __name__ == "__main__":
    fire.Fire(test_image)
