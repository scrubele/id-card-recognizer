import os

import cv2

from src import DEBUG_FOLDER
from src.data.preprocessors.matchers.feature.utils import make_distance_ratio_distribution, \
    extract_good_matches, apply_feature_detector, extract_good_matches_mask, detect_homography_polygon


def flann_matcher(template_image, input_image, image_name="", name="flann", algorithm_id=0):
    sift = cv2.SIFT_create()
    template_keypoints, template_descriptors, image_keypoints, image_descriptors = apply_feature_detector(
        feature_detector=sift, template_image=template_image, input_image=input_image)

    flann = cv2.FlannBasedMatcher(indexParams=dict(algorithm=algorithm_id, trees=5),  # FLAN_INDEX_KDTREE = 0
                                  searchParams=dict(checks=50))

    matches = flann.knnMatch(template_descriptors, image_descriptors, k=2)
    good_matches = extract_good_matches(matches, ratio=0.6)

    make_distance_ratio_distribution(matches, good_matches, name=image_name + "_" + name, display=False)
    good_flann_matches = cv2.drawMatchesKnn(
        template_image, template_keypoints, input_image, image_keypoints, good_matches, None, flags=2
    )
    # cv2.imshow("good_flann_matches", good_flann_matches)

    matches_mask = extract_good_matches_mask(matches, ratio=0.5)
    draw_params = dict(
        matchColor=(0, 0, 255),
        singlePointColor=(0, 255, 0),
        matchesMask=matches_mask,
        flags=2,
    )
    flann_matches = cv2.drawMatchesKnn(
        template_image, template_keypoints, input_image, image_keypoints, matches, None, **draw_params
    )
    # cv2.imshow("flann_matches_mask", flann_matches)
    # cv2.waitKey(0)

    polygon_image, polygon_corners = detect_homography_polygon(input_image, template_image, template_keypoints,
                                                               image_keypoints, good_matches, min_match_count=5)
    cv2.imwrite(str(os.path.join(DEBUG_FOLDER, image_name + "_polygon_" + name + ".png")), polygon_image)

    # cv2.imshow("polygon_image", polygon_image)
    # cv2.waitKey(0)
    return polygon_corners
