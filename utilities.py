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
import time
import random

# Third-party modules
from scipy import ndimage
from skimage.segmentation import slic, mark_boundaries
from skimage.measure import EllipseModel
from skimage.future import graph
import numpy as np
from cv2 import cv2

# Local modules
from config import ROCK_MIN_SIZE


@contextmanager
def Timer(msg):
    print(msg)
    start = time.perf_counter()
    try:
        yield
    finally:
        print("%.4f ms" % ((time.perf_counter() - start) * 1000))


def save_labels(path: str, labels, convert=True):
    """
    Save labels generated from segmentation methods for further inspection.
    Labels can be converted to 0 ~ 255 range optionally.

    :param path: Save file path
    :param labels: Labels
    :param convert: Whether convert label to 0 ~ 255 range
    :return: None
    """
    if convert:
        res = cv2.normalize(labels, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    else:
        res = labels

    cv2.imwrite(path, res)


def weight_mean_color(graph, src, dst, n):
    """
    Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `dst` is already computed.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the `"weight"` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
    """

    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}


def merge_mean_color(graph, src, dst):
    """Callback called before merging two nodes of a mean color distance graph.

    This method computes the mean color of `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] / graph.nodes[dst]['pixel count'])


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


def point_perspective_transform(M: np.ndarray, point: np.ndarray) -> np.ndarray:
    """
    This method performs a perspective transformation on a point using a given matrix M.

    :param M: 3*3 perspective transformation matrix from cv2.getPerspectiveTransform().
    :param point: (2, ) numpy array.
    :return: (2, ) numpy array, representing transformed point (also in y, x format).
    """
    assert [M.shape[0], M.shape[1]] == [3, 3], 'Invalid transformation matrix'
    assert point.shape[0] == 2, 'Invalid point format'

    temp = np.array([[point]])

    dst = cv2.perspectiveTransform(temp, M)

    return np.ravel(dst)


def calculate_point_diff(pt1: np.ndarray, pt2: np.ndarray) -> float:
    return abs(np.sum(pt1 - pt2))


def points_clockwise_sort(points: np.ndarray) -> np.ndarray:
    """
    Given a set of 2D points, return clockwise-sorted points.
    :param points: A set of 2D points.
    :return: Clockwise sorted points.
    """
    assert points.shape[1] == 2, 'Points should be 2D'

    sorted_index = np.lexsort((points[:, 0], points[:, 1]))
    return points[sorted_index]


def test_perspective_transformation(M: np.ndarray, src: np.ndarray, dst: np.ndarray):
    """
    This method test a given transformation matrix between two set of points.

    :param M: 3*3 perspective transformation matrix from cv2.getPerspectiveTransform().
    :param src: n_points * 2 numpy array with each line representing the location of each point, in format [yi, xi].
    :param dst: Same format as src.
    :return: Matched point index pairs in format [(i, j)], average matching error; None if test failed .
    """
    assert [M.shape[0], M.shape[1]] == [3, 3], 'Invalid transformation matrix'

    src_dim, dst_dim, src_len, dst_len = src.shape[1], dst.shape[1], src.shape[0], dst.shape[0]
    assert [src_dim, dst_dim] == [2, 2], 'Invalid point set format'

    pt_pairs = []
    error = 0

    used_dst = []

    for i in range(src_len):
        src_sample_pt = src[i].astype(np.float32)
        assumed_true_pt = point_perspective_transform(M, src_sample_pt)

        min_error = 10000000  # Just a large number
        min_error_index = 0

        for j in range(dst_len):
            if j in used_dst:
                continue

            current_err = calculate_point_diff(assumed_true_pt, dst[j].astype(np.float32))

            if current_err < min_error:
                min_error = current_err
                min_error_index = j

                if current_err <= 0.005:
                    break

        if min_error <= 50:
            # Match succeed if min_error is smaller than a threshold value
            # More experiments needed to find a suitable threshold
            pt_pairs.append((i, min_error_index))
            used_dst.append(min_error_index)
            error += min_error

    if len(pt_pairs) < 4:
        return None

    error = error / len(pt_pairs)  # Calculate average error among matched points

    return pt_pairs, error


def slic_wrapper(image, n_segments=5000, compactness=30, thresh=65, visualize=False):
    with Timer('Segmenting...'):
        labels = slic(image, n_segments=n_segments, compactness=compactness, start_label=1)

    with Timer('Merging...'):
        rag_graph = graph.rag_mean_color(image, labels)

        merged_labels = graph.merge_hierarchical(labels, rag_graph, thresh=thresh, rag_copy=False, in_place_merge=True,
                                                 merge_func=merge_mean_color, weight_func=weight_mean_color)

    number_of_regions = len(np.unique(merged_labels))

    segmented_image = np.empty(image.shape, dtype=np.uint8)

    for index in range(number_of_regions):
        region_index = np.where(merged_labels == index)

        segmented_image[region_index] = np.average(image[region_index], axis=0)

    if visualize:
        label_img = mark_boundaries(image, labels)
        merged_label_img = mark_boundaries(image, merged_labels)
        vis = np.hstack((label_img, merged_label_img))
        cv2.imshow('res', vis)
        cv2.waitKey()

    return segmented_image, merged_labels, number_of_regions
