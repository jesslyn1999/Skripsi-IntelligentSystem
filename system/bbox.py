from __future__ import division as _division
from __future__ import print_function as _print_function

import os as _os
import os.path as _path
import numpy as np
import cv2 as _cv2
from PIL import ImageFont
import numpy as _np
from hashlib import md5 as _md5
from typing import List

_LOC = _path.realpath(_path.join(_os.getcwd(), _path.dirname(__file__)))

COLOR_NAME_TO_RGB = dict(
    navy=((0, 38, 63), (119, 193, 250)),
    blue=((0, 120, 210), (173, 220, 252)),
    aqua=((115, 221, 252), (0, 76, 100)),
    teal=((15, 205, 202), (0, 0, 0)),
    olive=((52, 153, 114), (25, 58, 45)),
    green=((0, 204, 84), (15, 64, 31)),
    yellow=((255, 216, 70), (103, 87, 28)),
    orange=((255, 125, 57), (104, 48, 19)),
    red=((255, 47, 65), (131, 0, 17)),
    maroon=((135, 13, 75), (239, 117, 173)),
    fuchsia=((246, 0, 184), (103, 0, 78)),
    purple=((179, 17, 193), (241, 167, 244)),
    black=((24, 24, 24), (220, 220, 220)),
    gray=((168, 168, 168), (0, 0, 0)),
    silver=((220, 220, 220), (0, 0, 0)),
    lime=((127, 255, 1), (53, 102, 0))
)

_COLOR_NAMES = list(COLOR_NAME_TO_RGB)

_DEFAULT_COLOR_NAME = "green"

_FONT_PATH = _os.path.join(_LOC, "Ubuntu-B.ttf")
_FONT_HEIGHT = 15
_FONT = ImageFont.truetype(_FONT_PATH, _FONT_HEIGHT)


def _rgb_to_bgr(color):
    return list(reversed(color))


def _color_image(image, font_color, background_color):
    return background_color + (font_color - background_color) * image / 255


def _get_label_image(init_text: str, font_color_tuple_bgr, background_color_tuple_bgr):
    split_texts = init_text.splitlines()

    text_images = []
    for text in split_texts:
        text_image = _FONT.getmask(text)
        shape = list(reversed(text_image.size))
        bw_image = np.array(text_image).reshape(shape)

        image = [
            _color_image(bw_image, font_color, background_color)[None, ...]
            for font_color, background_color
            in zip(font_color_tuple_bgr, background_color_tuple_bgr)
        ]
        text_images.append(np.concatenate(image).transpose(1, 2, 0))

    return text_images


def add(image, left, top, right, bottom, label=None, color=None, place_label: str = "top"):
    if type(image) is not _np.ndarray:
        raise TypeError("'image' parameter must be a numpy.ndarray")
    try:
        left, top, right, bottom = int(left), int(top), int(right), int(bottom)
    except ValueError:
        raise TypeError("'left', 'top', 'right' & 'bottom' must be a number")

    if label and type(label) is not str:
        raise TypeError("'label' must be a str")

    if label and not color:
        hex_digest = _md5(label.encode()).hexdigest()
        color_index = int(hex_digest, 16) % len(COLOR_NAME_TO_RGB)
        color = _COLOR_NAMES[color_index]

    if not color:
        color = _DEFAULT_COLOR_NAME

    if type(color) is not str:
        raise TypeError("'color' must be a str")

    if color not in COLOR_NAME_TO_RGB:
        msg = "'color' must be one of " + ", ".join(COLOR_NAME_TO_RGB)
        raise ValueError(msg)

    if place_label not in ["top", "bottom"]:
        msg = "'color' must be one of " + ", ".join(["top", "bottom"])
        raise ValueError(msg)

    colors = [_rgb_to_bgr(item) for item in COLOR_NAME_TO_RGB[color]]
    color, color_text = colors

    _cv2.rectangle(image, (left, top), (right, bottom), color, 2)

    if label:
        image_height, image_width, _ = image.shape

        label_images = _get_label_image(label, color_text, color)

        label_heights = [img.shape[0] for img in label_images]

        if place_label == "top":
            is_crossed = (top - sum(label_heights)) < 0
        else:
            is_crossed = (bottom + sum(label_heights)) >= image_height

        for label_image_idx, label_image in enumerate(label_images):
            label_height, label_width, _ = label_image.shape

            rectangle_height, rectangle_width = 1 + label_height, 1 + label_width

            if place_label == "top":
                rectangle_bottom = top - sum(label_heights[label_image_idx + 1:])
                rectangle_left = max(0, min(left - 1, image_width - rectangle_width))

                rectangle_top = rectangle_bottom - rectangle_height
                rectangle_right = rectangle_left + rectangle_width

                label_top = rectangle_top + 1

                if is_crossed:
                    rectangle_top = top + sum(label_heights[:label_image_idx])
                    rectangle_bottom = rectangle_top + label_height + 1

                    label_top = rectangle_top

                label_left = rectangle_left + 1
                label_bottom = label_top + label_height
                label_right = label_left + label_width

                rec_left_top = (rectangle_left, rectangle_top)
                rec_right_bottom = (rectangle_right, rectangle_bottom)

            else:
                rectangle_top = bottom + sum(label_heights[:label_image_idx])
                rectangle_left = max(0, min(left - 1, image_width - rectangle_width))

                rectangle_bottom = rectangle_top + rectangle_height
                rectangle_right = rectangle_left + rectangle_width

                label_bottom = rectangle_bottom - 1

                if is_crossed:
                    rectangle_bottom = bottom - sum(label_heights[label_image_idx + 1:])
                    rectangle_top = rectangle_bottom - label_height - 1

                    label_bottom = rectangle_bottom

                label_left = rectangle_left + 1
                label_top = label_bottom - label_height
                label_right = label_left + label_width

                rec_left_top = (rectangle_left, rectangle_top)
                rec_right_bottom = (rectangle_right, rectangle_bottom)

            _cv2.rectangle(image, rec_left_top, rec_right_bottom, color, -1)
            image[label_top:label_bottom, label_left:label_right, :] = label_image
