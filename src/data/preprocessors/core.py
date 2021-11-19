import os

import cv2
import fire

from src import TEST_FOLDER, TEMPLATES_FOLDER
from src import logger
from src.data.preprocessors.matchers.features.flann import flann_matcher
from src.data.preprocessors.transformers.croppers import clip_by_coordinates
from src.data.preprocessors.transformers.rotators import rotate_image
from src.data.preprocessors.utils import is_card_detected, correct_card
from src.utils.status_manager import Status


def process_image(template_image, input_image, image_name, template_name):
    polygon_corners = flann_matcher(template_image, input_image, image_name=image_name)
    if is_card_detected(polygon_corners):
        polygon_corners, is_correct = correct_card(polygon_corners)
        clipped_image = clip_by_coordinates(input_image, polygon_corners, name="")
        logger.log_image(image_name, stage_name="clipped", image=clipped_image, save=True)
        rotated_image, angle = rotate_image(clipped_image)
        logger.info(f"Angle: {angle}")
        logger.log_image(image_name, stage_name="rotated", image=rotated_image, save=True)
        status = Status.OK if is_correct else Status.PARTIAL
        return {}, status
    else:
        logger.warning("Id card is not detected")
        return {}, Status.NOT_FOUND


def preprocess_data(image_path, template_path=str(os.path.join(TEMPLATES_FOLDER, "template.jpg"))):
    image_name = os.path.basename(image_path)
    template_name = os.path.basename(image_path)
    try:
        template_image = cv2.imread(template_path)
        input_image = cv2.imread(image_path)
        template_image = cv2.resize(template_image, None, fx=0.5, fy=0.5)
        input_image = cv2.resize(input_image, None, fx=0.5, fy=0.5)
    except:
        logger.warning("Input image or template image are not found")
    info, status = process_image(template_image, input_image, image_name, template_name)
    cv2.destroyAllWindows()
    return info, status


def test_image(image_name="1.jpg"):
    image_path = str(os.path.join(TEST_FOLDER, image_name))
    preprocess_data(image_path)


if __name__ == "__main__":
    fire.Fire(test_image)
