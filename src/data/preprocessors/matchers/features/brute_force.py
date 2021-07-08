import os

import cv2
from matplotlib import pyplot as plt

from src import DEBUG_FOLDER
from src.data.preprocessors.matchers.features.utils import calculate_distance_distribution, \
    make_distance_ratio_distribution, \
    extract_good_matches, apply_feature_detector, detect_homography_polygon


def brute_force_matching_with_orb(template_image, input_image, image_name="", name="_brute_force_orb"):
    orb = cv2.ORB_create()  # ORB detector
    template_keypoints, template_descriptors, image_keypoints, image_descriptors = apply_feature_detector(
        feature_detector=orb, template_image=template_image, input_image=input_image)

    brute_force_matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
    matches = brute_force_matcher.match(template_descriptors, image_descriptors)
    matches = sorted(matches, key=lambda x: x.distance)

    distances, prob_distribution = calculate_distance_distribution(matches)
    plt.plot(distances, prob_distribution)
    # plt.savefig(str(os.path.join(DEBUG_FOLDER, image_name + name + "_distance_ratio.png")))
    # plt.show()

    orb_matches = cv2.drawMatches(
        template_image, template_keypoints, input_image, image_keypoints, matches[:50], None, flags=2
    )
    # cv2.imshow("brute_force_matching_with_orb", orb_matches)
    # cv2.waitKey(0)

    polygon_image, polygon_corners = detect_homography_polygon(input_image, template_image, template_keypoints,
                                                               image_keypoints, matches, min_match_count=5)
    cv2.imwrite(str(os.path.join(DEBUG_FOLDER, image_name + "_polygon_" + name + ".png")), polygon_image)

    # cv2.imshow("polygon_image", polygon_image)
    # cv2.waitKey(0)
    return polygon_corners


def brute_force_matching_with_sift(template_image, input_image, image_name="", name="bruto_force_sift"):
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
    # cv2.imshow("brute_force_matching_with_sift", sift_matches)
    # cv2.waitKey(0)

    polygon_image, polygon_corners = detect_homography_polygon(input_image, template_image, template_keypoints,
                                                               image_keypoints, good_matches, min_match_count=5)
    cv2.imwrite(str(os.path.join(DEBUG_FOLDER, image_name + "_polygon_" + name + ".png")), polygon_image)

    # cv2.imshow("polygon_image", polygon_image)
    # cv2.waitKey(0)
    return polygon_corners
