"""
THIS REPO IS STILL IN ACTIVE DEVELOPMENT.

This script contains some methods used in main function

Copyleft Lang Zhou, zhoulang731@tongji.edu.cn

GitHub: https://github.com/Mars-Rover-Localization/RockMatching

Created May 2021

Last modified July 2021
"""

# Built-in modules
import copy
import math
from contextlib import contextmanager

# Third-party modules
import numpy as np
from scipy import ndimage
from skimage.measure import EllipseModel
import cv2

# Local modules
from config import ROCK_MIN_SIZE


def clock():
    return cv2.getTickCount() / cv2.getTickFrequency()


@contextmanager
def Timer(msg):
    print(msg)
    start = clock()
    try:
        yield
    finally:
        print("%.4f ms" % ((clock() - start) * 1000))


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


def ellipse_sparsing(ellipses):
    current_number = len(ellipses[0])

    while True:
        flag = False
        new_ellipses = [[], [], []]

        for i in range(len(ellipses[1])):
            for j in range(i + 1, len(ellipses[1])):
                ellipse_x = ellipses[1][i]
                ellipse_y = ellipses[1][j]

                distance = math.sqrt((ellipse_x[0] - ellipse_y[0]) ** 2 + (ellipse_x[1] - ellipse_y[1]) ** 2)
                threshold = math.sqrt(ellipse_x[2] ** 2 + ellipse_x[3] ** 2) + math.sqrt(ellipse_y[2] ** 2 + ellipse_y[3] ** 2)

                if abs(distance / threshold - 1) <= 0.45:
                    new_edge = ellipses[2][i] + ellipses[2][j]
                    edge_points = np.transpose(np.nonzero(new_edge))

                    params = ellipse_model_fitting(edge_points)

                    if params == -1:
                        continue

                    new_size = math.pi * params[2] * params[3]

                    new_ellipses[0].append(new_size)
                    new_ellipses[1].append(list(params))
                    new_ellipses[2].append(new_edge)
                else:   # TODO: Handle the duplicate insertion problem
                    new_ellipses[0].append(ellipses[0][i])
                    new_ellipses[1].append(ellipses[1][i])
                    new_ellipses[2].append(ellipses[2][i])
                    new_ellipses[0].append(ellipses[0][j])
                    new_ellipses[1].append(ellipses[1][j])
                    new_ellipses[2].append(ellipses[2][j])


        if not flag:
            break
