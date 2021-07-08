import os

import cv2
import fire

from src import TEST_FOLDER, TEMPLATES_FOLDER
from src.data.preprocessors.matchers.features.core import process_feature_mapping_approaches
from src.data.preprocessors.matchers.templates.core import process_template_matching_method_comparison


def preprocess_data(image_path, template_path=str(os.path.join(TEMPLATES_FOLDER, "template.jpg"))):
    image_name = os.path.basename(image_path)
    template_image = cv2.imread(template_path)
    input_image = cv2.imread(image_path)
    template_image = cv2.resize(template_image, None, fx=0.5, fy=0.5)
    input_image = cv2.resize(input_image, None, fx=0.5, fy=0.5)

    process_feature_mapping_approaches(template_image, input_image, image_name=image_name)
    process_template_matching_method_comparison(template_image, input_image, image_name=image_name)
    cv2.destroyAllWindows()


def test_image(image_name="1.jpg"):
    image_path = str(os.path.join(TEST_FOLDER, image_name))
    preprocess_data(image_path)


if __name__ == "__main__":
    fire.Fire(test_image)
