from copy import deepcopy


def is_card_detected(polygon_corners):
    return True if len(polygon_corners) > 0 else False


def correct_card(polygon_corners, threshold=-1):
    negative_values = deepcopy(polygon_corners).ravel()
    negative_values = negative_values[negative_values < threshold]
    negative_values[negative_values < threshold] = 0
    is_correct = False if len(negative_values) > 0 else True
    polygon_corners[polygon_corners < 0] = 0
    return polygon_corners, is_correct
