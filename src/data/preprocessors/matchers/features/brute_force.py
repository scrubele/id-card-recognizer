import cv2
from matplotlib import pyplot as plt

from src import logger
from src.data.preprocessors.matchers.features.utils import calculate_distance_distribution, \
    make_distance_ratio_distribution, \
    extract_good_matches, apply_feature_detector, detect_homography_polygon


def brute_force_matching_with_orb(template_image, input_image, image_name="", func_name="bf_orb"):
    orb = cv2.ORB_create()  # ORB detector
    template_keypoints, template_descriptors, image_keypoints, image_descriptors = apply_feature_detector(
        feature_detector=orb, template_image=template_image, input_image=input_image)

    brute_force_matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
    matches = brute_force_matcher.match(template_descriptors, image_descriptors)
    matches = sorted(matches, key=lambda x: x.distance)

    distances, prob_distribution = calculate_distance_distribution(matches)
    plt.plot(distances, prob_distribution)
    logger.log_plot(plt, image_name, stage_name=func_name + "_prob_distribution", save=True)

    orb_matches = cv2.drawMatches(
        template_image, template_keypoints, input_image, image_keypoints, matches[:50], None, flags=2
    )
    logger.log_image(image_name, stage_name=func_name + "_matches", image=orb_matches, save=True)

    polygon_image, polygon_corners = detect_homography_polygon(input_image, template_image, template_keypoints,
                                                               image_keypoints, matches, min_match_count=5)
    logger.log_image(image_name, stage_name=func_name + "_polygon_image", image=polygon_image, save=True)
    return polygon_corners


def brute_force_matching_with_sift(template_image, input_image, image_name="", name="bf_sift"):
    sift = cv2.SIFT_create()
    template_keypoints, template_descriptors, image_keypoints, image_descriptors = apply_feature_detector(
        feature_detector=sift, template_image=template_image, input_image=input_image)

    brute_force_matcher = cv2.BFMatcher()
    matches = brute_force_matcher.knnMatch(template_descriptors, image_descriptors, k=2)
    good_matches = extract_good_matches(matches, ratio=0.6)

    make_distance_ratio_distribution(matches, good_matches, name=image_name + "_" + name)

    sift_matches = cv2.drawMatchesKnn(
        template_image, template_keypoints, input_image, image_keypoints, good_matches, None, flags=2
    )
    logger.log_image(image_name, stage_name=name + "_matches", image=sift_matches, save=True)

    polygon_image, polygon_corners = detect_homography_polygon(input_image, template_image, template_keypoints,
                                                               image_keypoints, good_matches, min_match_count=5)
    logger.log_image(image_name, stage_name=name + "_polygon_image", image=polygon_image, save=True)

    return polygon_corners
