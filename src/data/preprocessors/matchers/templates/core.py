import os

import cv2
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from src import DEBUG_FOLDER
from src.data.preprocessors.matchers.templates.utils import apply_template_matching_method

TEMPLATE_MATCHING_METHODS = ["cv2.TM_CCOEFF", "cv2.TM_CCOEFF_NORMED", "cv2.TM_CCORR",
                             "cv2.TM_CCORR_NORMED", "cv2.TM_SQDIFF", "cv2.TM_SQDIFF_NORMED", ]


def apply_template_matching(input_img, method_name, template):
    return apply_template_matching_method(input_img, method_name, template)


def process_template_matching_method_comparison(template, input_img, image_name, template_name="template.jpg"):
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    methods = TEMPLATE_MATCHING_METHODS
    fig = plt.figure(constrained_layout=True, figsize=(16, 9))
    gs = GridSpec(ncols=len(methods), nrows=2, wspace=0.0, hspace=0.0, figure=fig)
    for i, method in enumerate(methods):
        result, polygon_image, _ = apply_template_matching(input_img, method, template)
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(result, cmap="gray")
        ax.set_title(method)
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(cv2.cvtColor(polygon_image, cv2.COLOR_BGR2RGB))
    fig.savefig(str(os.path.join(DEBUG_FOLDER, image_name + "_template_comparison_plot.png")))
