"""
THIS REPO IS STILL IN ACTIVE DEVELOPMENT.

This script contains some methods used in main function

Copyleft Lang Zhou, zhoulang731@tongji.edu.cn

GitHub: https://github.com/Mars-Rover-Localization/RockMatching

Created May 2021

Last modified June 2021
"""

# Built-in modules
import math

# Third-party modules
import numpy as np
from scipy import ndimage
from skimage.measure import EllipseModel

# Local modules
from config import ROCK_MIN_SIZE


def edge_extraction(area):
    erode = ndimage.binary_erosion(area).astype(area.dtype)
    edge = area ^ erode
    return edge


def ellipse_model_fitting(points):
    ellipse = EllipseModel()

    if not ellipse.estimate(points):
        return -1

    xc, yc, a, b, theta = ellipse.params

    if a * b == 0:
        return -1

    c = math.sqrt(abs(a ** 2 - b ** 2))
    e = c / a

    return xc, yc, a, b, c, e, theta


def ellipse_filtering(current_index, params):
    if params == -1:
        print(f"Ellipse fitting failed for area {current_index}")
        return False

    xc, yc, a, b, c, e, theta = params

    ellipse_area = math.pi * a * b

    if ellipse_area < ROCK_MIN_SIZE * 0.5 or ellipse_area > ROCK_MIN_SIZE ** 2:
        print(f"Ellipse shape rejected for area {current_index}, low fitting ratio")
        return False

    if e > 0.99:
        print(f"Ellipse shape rejected for area {current_index}, too large eccentricity")
        return False

    return params
