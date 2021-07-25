"""
THIS REPO IS STILL IN ACTIVE DEVELOPMENT.

This script contains some methods used in main function

Copyleft Lang Zhou, zhoulang731@tongji.edu.cn

GitHub: https://github.com/Mars-Rover-Localization/RockMatching

Created May 2021

Last modified July 2021
"""

# Built-in modules
import math
from contextlib import contextmanager

# Third-party modules
from scipy import ndimage
from skimage.measure import EllipseModel
import numpy as np
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
        # print(f"Ellipse fitting failed for area {current_index}")
        return False

    xc, yc, a, b, c, e, theta = params

    ellipse_area = math.pi * a * b

    if ellipse_area < ROCK_MIN_SIZE * 0.5 or ellipse_area > ROCK_MIN_SIZE ** 2:
        # print(f"Ellipse shape rejected for area {current_index}, low fitting ratio")
        return False

    if e > 0.99:
        # print(f"Ellipse shape rejected for area {current_index}, too large eccentricity")
        return False

    return params


def ellipse_sparsing(ellipses):
    new_ellipses = []
    used_ellipses = []

    for i in range(len(ellipses)):
        for j in range(i + 1, len(ellipses)):
            ellipse_x = ellipses[i]
            ellipse_y = ellipses[j]

            distance = math.sqrt((ellipse_x[1][0] - ellipse_y[1][0]) ** 2 + (ellipse_x[1][1] - ellipse_y[1][1]) ** 2)
            threshold = math.sqrt(ellipse_x[1][2] ** 2 + ellipse_x[1][3] ** 2) + math.sqrt(
                ellipse_y[1][2] ** 2 + ellipse_y[1][3] ** 2)

            if abs(distance / threshold - 1) <= 0.45:
                new_edge = ellipse_x[2] + ellipse_y[2]
                edge_points = np.transpose(np.nonzero(new_edge))

                params = ellipse_model_fitting(edge_points)

                if params == -1:
                    new_size = ellipse_x[0] + ellipse_y[0]
                else:
                    new_size = math.pi * params[2] * params[3]

                new_ellipses.append([new_size, list(params), new_edge])
                used_ellipses.append(ellipse_x)
                used_ellipses.append(ellipse_y)
            else:
                temp = [x[0] for x in new_ellipses] + [x[0] for x in used_ellipses]
                if ellipse_x[0] not in temp:
                    new_ellipses.append(ellipse_x)

                if ellipse_y[0] not in temp:
                    new_ellipses.append(ellipse_y)

    temp = [x[0] for x in used_ellipses]
    result = [x for x in new_ellipses if x[0] not in temp]

    return result


def visualize_rocks(img, rocks):
    for rock in rocks:
        edge = rock[2]
        edge_points = np.transpose(np.nonzero(edge))
        img[tuple(edge_points.transpose())] = [0, 0, 255]
        position = (int(rock[1][1]), int(rock[1][0]))
        img = cv2.putText(img, str(int(rock[0])), position, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)
    return img


def point_perspective_transform(M: np.ndarray, point: np.ndarray):
    """
    This method performs a perspective transformation on a point using a given matrix M.

    :param M: 3*3 perspective transformation matrix from cv2.getPerspectiveTransform()
    :param point: (2, ) numpy array
    :return: 1*2 numpy array, representing transformed point (also in y, x format)
    """
    assert [M.shape[0], M.shape[1]] == [3, 3], 'Invalid transformation matrix'
    assert point.shape[0] == 2, 'Invalid point format'

    temp = M @ np.array([[point[0]], [point[1]], [1]])
    print(temp)
    dst = temp[0:2, :].transpose() / temp[2, 0]

    return dst
