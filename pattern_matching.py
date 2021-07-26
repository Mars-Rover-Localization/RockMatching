import cv2
import numpy as np
import random
import utilities as utl


def find_corner_point(area, identifier):
    """
    This method is part of a dismissed matching idea, could be deprecated in the future.

    :param area: numpy.ndarray, area containing interest points
    :param identifier: int, specific value distinguishing interest points from other pixels
    :return: numpy.ndarray, corner point's position, in format of [x, y]
    """
    mask = (area == identifier)
    coordinates = np.nonzero(mask)

    distance = [coordinates[0][index] ** 2 + coordinates[1][index] ** 2 for index in range(len(coordinates[0]))]
    min_distance_index = distance.index(min(distance))

    return np.transpose(coordinates)[min_distance_index]


def perspective_based_random_matching(src_points: np.ndarray, dst_points: np.ndarray, iteration: int):
    """
    This method matches two set of points using an algorithm we named Perspective Based Random Matching.

    For consistency with cv2.getPerspectiveTransform(), all points in this method is formatted as [col, row].

    For more details, please refer to our future paper.

    :param src_points: n_points * 2 numpy array with each line representing the location of each point, in format [yi, xi].
    :param dst_points: Same format as src_points
    :param iteration: Iteration times. Larger iteration times may improve the precision of the result, while taking more computing time.
    :return: Perspective transformation matrix
    """
    assert iteration > 0, 'Invalid iteration parameter'

    src_len, dst_len = src_points.shape[0], dst_points.shape[0]
    assert [src_len, dst_len] >= [4, 4], 'Too few points for perspective estimation'

    M = np.zeros((3, 3))
    min_error = 10000000
    matched_pts = []
    flag = False

    for _ in range(iteration):
        src_random_pts = src_points[random.sample(range(src_len), 4)].astype(np.float32)

        for _ in range(iteration):
            dst_random_pts = dst_points[random.sample(range(dst_len), 4)].astype(np.float32)

            current_M = cv2.getPerspectiveTransform(src_random_pts, dst_random_pts)

            try:
                pt_pairs, error = utl.test_perspective_transformation(current_M, src_points, dst_points)
            except TypeError:
                continue

            if error < min_error:
                flag = True

                M = current_M
                min_error = error
                matched_pts = pt_pairs

    if not flag:
        return None

    return M, matched_pts, min_error
