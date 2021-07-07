import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import norm

from src import DATA_FOLDER, TEST_FOLDER, DEBUG_FOLDER


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
    distances = sorted([get_match(match_list).distance for match_list in matches])
    probability_distribution = norm.cdf(distances, np.mean(distances), np.std(distances))
    return distances, probability_distribution


def make_distance_ratio_distribution(matches, good_matches, name="", display=False):
    distances, prob_distribution = calculate_distance_distribution(matches)
    good_distances, good_prob_distribution = calculate_distance_distribution(good_matches)

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.plot(distances, prob_distribution)
    ax.plot(good_distances, good_prob_distribution)
    ax.set_title(name)
    if display:
        plt.show()
    fig.savefig(str(os.path.join(DEBUG_FOLDER, name + "_distance_ratio.png")))


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
        box_corners = cv2.perspectiveTransform(obj_corners, M)

        cv2.polylines(img, [np.int32(box_corners)], True, 255, 3, cv2.LINE_AA)
        return img
    else:
        print("Not enough matches are found - {}/{}".format(len(good_matches), min_match_count))
        return img


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

    polygon_image = detect_homography_polygon(input_image, template_image, template_keypoints, image_keypoints, matches,
                                              min_match_count=5)
    cv2.imwrite(str(os.path.join(DEBUG_FOLDER, image_name + "_polygon_" + name + ".png")), polygon_image)

    # cv2.imshow("polygon_image", polygon_image)
    # cv2.waitKey(0)


def bruto_force_matching_with_sift(template_image, input_image, image_name="", name="bruto_force_sift"):
    sift = cv2.xfeatures2d.SIFT_create()
    template_keypoints, template_descriptors, image_keypoints, image_descriptors = apply_feature_detector(
        feature_detector=sift, template_image=template_image, input_image=input_image)

    brute_force_matcher = cv2.BFMatcher()
    matches = brute_force_matcher.knnMatch(template_descriptors, image_descriptors, k=2)
    good_matches = extract_good_matches(matches, ratio=0.6)

    make_distance_ratio_distribution(matches, good_matches, name=image_name + "_" + name)

    sift_matches = cv2.drawMatchesKnn(
        template_image, template_keypoints, input_image, image_keypoints, good_matches, None, flags=2
    )
    # cv2.imshow("bruto_force_matching_with_sift", sift_matches)
    # cv2.waitKey(0)

    polygon_image = detect_homography_polygon(input_image, template_image, template_keypoints, image_keypoints,
                                              good_matches,
                                              min_match_count=5)
    cv2.imwrite(str(os.path.join(DEBUG_FOLDER, image_name + "_polygon_" + name + ".png")), polygon_image)
    # cv2.imshow("polygon_image", polygon_image)
    # cv2.waitKey(0)


def flann_matcher(template_image, input_image, image_name="", name="flann"):
    sift = cv2.xfeatures2d.SIFT_create()
    template_keypoints, template_descriptors, image_keypoints, image_descriptors = apply_feature_detector(
        feature_detector=sift, template_image=template_image, input_image=input_image)

    FLAN_INDEX_KDTREE = 0
    flann = cv2.FlannBasedMatcher(indexParams=dict(algorithm=FLAN_INDEX_KDTREE, trees=5),
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

    polygon_image = detect_homography_polygon(input_image, template_image, template_keypoints, image_keypoints,
                                              good_matches,
                                              min_match_count=5)
    cv2.imwrite(str(os.path.join(DEBUG_FOLDER, image_name + "_polygon_" + name + ".png")), polygon_image)
    # cv2.imshow("polygon_image", polygon_image)
    # cv2.waitKey(0)


# TEMPLATE MATCHING METHODS

TEMPLATE_MATCHING_METHODS = ["cv2.TM_CCOEFF", "cv2.TM_CCOEFF_NORMED", "cv2.TM_CCORR",
                             "cv2.TM_CCORR_NORMED", "cv2.TM_SQDIFF", "cv2.TM_SQDIFF_NORMED", ]


def apply_template_matching_method(input_img, method_name, template):
    current_input_img = input_img.copy()
    img = cv2.cvtColor(current_input_img, cv2.COLOR_BGR2GRAY)
    method_name = eval(method_name)
    result = cv2.matchTemplate(img, template, method_name)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if method_name in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + template.shape[0], top_left[1] + template.shape[1])
    polygon_image = cv2.rectangle(
        current_input_img, top_left, bottom_right, color=(0, 0, 255), thickness=2
    )
    return result, polygon_image


# APPROACHES

def process_template_matching_method_comparison(input_img, template, image_name, template_name="template.jpg"):
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    methods = TEMPLATE_MATCHING_METHODS
    fig = plt.figure(constrained_layout=True, figsize=(16, 9))
    gs = GridSpec(ncols=len(methods), nrows=2, wspace=0.0, hspace=0.0, figure=fig)
    for i, method in enumerate(methods):
        result, polygon_image = apply_template_matching_method(input_img, method, template)
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(result, cmap="gray")
        ax.set_title(method)
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(cv2.cvtColor(polygon_image, cv2.COLOR_BGR2RGB))
    fig.savefig(str(os.path.join(DEBUG_FOLDER, image_name + "_template_comparison_plot.png")))


def process_feature_mapping_approaches(template_image, input_image, image_name="1.jpg", template_name="template.jpg"):
    template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
    input_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # cv2.imshow("template_gray", template_gray)
    # cv2.imshow("input_gray", input_gray)
    # cv2.waitKey(0)
    brute_force_matching_with_orb(template_image, input_image, image_name=image_name)
    bruto_force_matching_with_sift(template_image, input_image, image_name=image_name)
    flann_matcher(template_image, input_image, image_name=image_name)


def process_fies():
    for i in range(15):
        image_name = str(i) + ".jpg"
        print(image_name)
        process_feature_mapping_approaches(image_name)


def preprocess_data(image_path, template_path=str(os.path.join(DATA_FOLDER, "template.jpg"))):
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
        test_image()
