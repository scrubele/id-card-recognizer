import cv2

from src import logger
from src.data.preprocessors.matchers.features.utils import make_distance_ratio_distribution, \
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
    logger.log_image(image_name, stage_name="flann_good_matches", image=good_flann_matches, save=True)

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
    logger.log_image(image_name, stage_name="flann_matches", image=flann_matches, save=True)

    polygon_image, polygon_corners = detect_homography_polygon(input_image, template_image, template_keypoints,
                                                               image_keypoints, good_matches, min_match_count=5)
    logger.log_image(image_name, stage_name="flann_polygons", image=polygon_image, save=True)
    return polygon_corners
