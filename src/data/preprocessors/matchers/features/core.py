import os

import fire

from src import TEST_FOLDER
from src.data.preprocessors.matchers.features.brute_force import brute_force_matching_with_orb, \
    brute_force_matching_with_sift
from src.data.preprocessors.matchers.features.flann import flann_matcher


def process_feature_mapping_approaches(template_image, input_image, image_name="1.jpg", template_name="template.jpg"):
    brute_force_matching_with_orb(template_image, input_image, image_name=image_name)
    brute_force_matching_with_sift(template_image, input_image, image_name=image_name)
    flann_matcher(template_image, input_image, image_name=image_name)


def test_image(image_name="1.jpg"):
    image_path = str(os.path.join(TEST_FOLDER, image_name))
    process_feature_mapping_approaches(image_path)


if __name__ == "__main__":
    fire.Fire(test_image)
