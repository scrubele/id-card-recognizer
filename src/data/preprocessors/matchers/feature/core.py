from src.data.preprocessors.matchers.feature.brute_force import brute_force_matching_with_orb, \
    brute_force_matching_with_sift
from src.data.preprocessors.matchers.feature.flann import flann_matcher


def process_feature_mapping_approaches(template_image, input_image, image_name="1.jpg", template_name="template.jpg"):
    brute_force_matching_with_orb(template_image, input_image, image_name=image_name)
    brute_force_matching_with_sift(template_image, input_image, image_name=image_name)
    flann_matcher(template_image, input_image, image_name=image_name)
