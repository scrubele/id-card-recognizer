import cv2


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
    polygon_coordinates = [top_left, bottom_right]
    return result, polygon_image, polygon_coordinates
