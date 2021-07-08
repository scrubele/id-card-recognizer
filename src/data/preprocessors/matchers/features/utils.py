import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm

from src import logger


def apply_feature_detector(feature_detector, template_image, input_image):
    template_keypoints, template_descriptors = feature_detector.detectAndCompute(template_image, None)
    image_keypoints, image_descriptors = feature_detector.detectAndCompute(input_image, None)
    return template_keypoints, template_descriptors, image_keypoints, image_descriptors


def extract_good_matches(matches, ratio=0.6):
    good_matches = []
    for m1, m2 in matches:
        if m1.distance < ratio * m2.distance:
            good_matches.append([m1])
    return good_matches


def extract_good_matches_mask(matches, ratio=0.5):
    matches_mask = np.zeros((len(matches), 2))
    for i, (m1, m2) in enumerate(matches):
        if m1.distance < ratio * m2.distance:
            matches_mask[i] = [1, 0]
    return matches_mask


def get_match(match, dimension=0):
    if isinstance(match, list) or isinstance(match, np.ndarray):
        return match[0]
    else:
        return match


def calculate_distance_distribution(matches):
    np.seterr(all="raise")
    distances = sorted([get_match(match_list).distance for match_list in matches])
    try:
        probability_distribution = norm.cdf(distances, np.mean(distances), np.std(distances))
    except Exception:
        logger.warning("Probability distribution encountered mean of empty slice")
        probability_distribution = []
    return distances, probability_distribution


def make_distance_ratio_distribution(matches, good_matches, name="", display=False):
    distances, prob_distribution = calculate_distance_distribution(matches)
    good_distances, good_prob_distribution = calculate_distance_distribution(good_matches)
    if len(good_distances) > 0 and len(good_prob_distribution) > 0:
        fig, ax = plt.subplots(figsize=(11, 8))
        ax.plot(distances, prob_distribution)
        ax.plot(good_distances, good_prob_distribution)
        ax.set_title(name)
        if display:
            plt.show()
        logger.log_figure(fig, name + "_template_comparison_plot")


def draw_polygon(img, polygon_corners):
    cv2.polylines(img, [np.int32(polygon_corners)], True, 255, 3, cv2.LINE_AA)
    return img


def detect_homography_polygon(input_image, template_image, template_keypoints, image_keypoints, good_matches,
                              min_match_count=5,
                              dimension=2):
    img = input_image.copy()
    if len(good_matches) > min_match_count:
        source_points = np.float32(
            [template_keypoints[get_match(m, dimension).queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        destination_points = np.float32(
            [image_keypoints[get_match(m, dimension).trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(source_points, destination_points, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()

        h, w, d = template_image.shape
        obj_corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        polygon_corners = cv2.perspectiveTransform(obj_corners, M)

        img = draw_polygon(img, polygon_corners)
        return img, polygon_corners
    else:
        logger.warning("Not enough matches are found - {}/{}".format(len(good_matches), min_match_count))
        return img, []
