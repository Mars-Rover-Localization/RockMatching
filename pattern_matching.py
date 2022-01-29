from cv2 import cv2
import numpy as np
import utilities as utl

import random
import math
import matplotlib.pyplot as plt
from ICP.icp import icp, point_based_matching

"""
Last edited January 2022: Added Iterative Closest Point based matching method, this algorithm has huge advantage over our previous methods.

This ICP algorithm is originally implemented by @richardos at GitHub (https://github.com/richardos/icp), for more copyright info, please refer to ICP/LICENSE
"""


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
        src_random_pts = utl.points_clockwise_sort(src_random_pts)

        for _ in range(iteration):
            dst_random_pts = dst_points[random.sample(range(dst_len), 4)].astype(np.float32)
            dst_random_pts = utl.points_clockwise_sort(dst_random_pts)

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


def icp_wrapper(reference_points: np.ndarray, points_to_be_aligned: np.ndarray, verbose=False):
    assert reference_points.shape[1] == 2 and points_to_be_aligned.shape[1] == 2, 'Invalid data shape'

    transformation_history, aligned_points = icp(reference_points, points_to_be_aligned, verbose=verbose)

    registration_result = []

    for index in range(len(aligned_points)):
        registration_result.append((points_to_be_aligned[index], aligned_points[index]))

    rotation, t_x, t_y = point_based_matching(registration_result)

    return rotation, t_x, t_y


def icp_example():
    np.random.seed(12345)

    # create a set of points to be the reference for ICP
    xs = np.random.random_sample((50, 1))
    ys = np.random.random_sample((50, 1))
    reference_points = np.hstack((xs, ys))

    # transform the set of reference points to create a new set of
    # points for testing the ICP implementation

    # 1. remove some points
    points_to_be_aligned = reference_points[1:47]

    # 2. apply rotation to the new point set
    theta = math.radians(12)
    c, s = math.cos(theta), math.sin(theta)
    rot = np.array([[c, -s],
                    [s, c]])
    points_to_be_aligned = np.dot(points_to_be_aligned, rot)

    # 3. apply translation to the new point set
    true_tx, true_ty = np.random.random_sample(), np.random.random_sample()
    points_to_be_aligned += np.array([true_tx, true_ty])

    # run icp
    transformation_history, aligned_points = icp(reference_points, points_to_be_aligned, verbose=True)

    registration_result = []

    for index in range(len(aligned_points)):
        registration_result.append((points_to_be_aligned[index], aligned_points[index]))

    rotation, t_x, t_y = point_based_matching(registration_result)

    print("True transformation:")
    print(theta, true_tx, true_ty)

    print("Estimated transformation:")
    print(rotation, t_x, t_y)

    # show results
    plt.plot(reference_points[:, 0], reference_points[:, 1], 'rx', label='reference points')
    plt.plot(points_to_be_aligned[:, 0], points_to_be_aligned[:, 1], 'b1', label='points to be aligned')
    plt.plot(aligned_points[:, 0], aligned_points[:, 1], 'g+', label='aligned points')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    src = np.array([[0, 0], [10, 0], [10, 5], [0, 5]])
    dst = np.array([[0, 0], [5, 0], [5, 2.5], [0, 2.5]])
    print(perspective_based_random_matching(src, dst, 50))
