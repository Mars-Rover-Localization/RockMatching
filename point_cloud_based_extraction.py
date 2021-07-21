"""
This file implements the basic idea of extracting rocks from point cloud.

Since no field test has been conducted due to the lack of stereo camera, the code currently only serve as demonstration.

Copyleft Lang Zhou, zhoulang731@tongji.edu.cn

GitHub: https://github.com/Mars-Rover-Localization/RockMatching

Created July 2021

Last modified July 2021
"""

# Third-party modules
import numpy as np
from sklearn.neighbors import KDTree

# Local modules
import utilities as utl


def plane_fitting(points: np.ndarray):
    # TODO: Inspect the necessity of using stereo-cam generated X-Y coordinates
    """
    Fit a plane from point cloud, using least squares algorithm.
    Performance haven't been evaluated under large input, alternative methods such as SVD maybe inspected later.

    :param points: numpy.ndarray matrix, with each row consisting of [x, y, z], total size is n * 3.
    :return: parameter a, b, c of plane equation ax + by + c = z.
    """
    assert points.shape[1] == 3, 'Dimension of points should be 3'
    n = points.shape[0]
    A = np.hstack((points[:, 0:2], np.ones((n, 1))))
    B = points[:, 2]

    x = np.linalg.inv((A.transpose() @ A)) @ A.transpose() @ B
    a, b, c = x.transpose().tolist()

    return a, b, c


def calculate_diff(points: np.ndarray, terrain_equation: tuple):
    """
    Calculate the difference between point cloud data and fitted terrain plane.

    :param points: n_points * 3 numpy array.
    :param terrain_equation: tuple (a, b, c) from utl.plane_fitting()
    :return: n_points * 1 numpy array with each element representing the corresponding difference.
    """
    assert len(terrain_equation) == 3, 'Incorrect terrain plane equation format'

    a, b, c = terrain_equation
    return np.fromfunction(lambda i, j: points[i, 2] - (a * points[i, 0] + b * points[i, 1] + c), (points.shape[0], 1), dtype=float)


def locally_extreme_points(coords: np.ndarray, data: np.ndarray, threshold_abs: float, neighbourhood: int = 10,
                           lookfor='max'):
    """
    Find local maxima of points in a point cloud.
    Modified based on a StackOverflow answer: https://stackoverflow.com/a/27116281/12524821

    :param coords: A shape (n_points, n_dims) array of point locations.
    :param data: A shape (n_points, ) vector of point values.
    :param threshold_abs: Minimum value to be considered as a maxima.
    :param neighbourhood: The (scalar) size of the neighbourhood in which to search.
    :param lookfor: Either 'max', or 'min', depending on whether you want local maxima or minima.

    :returns:
        filtered_coords: The coordinates of locally extreme points.
        filtered_data: The values of these points.
    """
    assert coords.shape[0] == data.shape[0], 'Number of points height data and coordinates data should be equal'
    assert neighbourhood >= 1, 'Searching neighborhood radius should be a positive integer'

    extreme_fcn = {'min': np.min, 'max': np.max}[lookfor]
    kdtree = KDTree(coords)
    neighbours = kdtree.query_radius(coords, r=neighbourhood)

    extreme = [data[i] == extreme_fcn(data[n]) and data[i] >= threshold_abs for i, n in enumerate(neighbours)]
    extrema = np.nonzero(extreme)   # extrema will contain a row of empty zeros since extreme is a 1 * n list

    return coords[extrema[0], 0:2], data[extrema[0]]


# Plane fitting test data
x = np.array([[0.274791784, -1.001679346, -1.851320839, 0.365840754]])
y = np.array([[-1.155674199, -1.215133985, 0.053119249, 1.162878076]])
z = np.array([[1.216239624, 0.764265677, 0.956099579, 1.198231236]])
test_points = np.hstack((x.transpose(), y.transpose(), z.transpose()))

with utl.Timer("Plane fitting..."):
    a, b, c = plane_fitting(test_points)

print(f"Fitted plane: {a}x + {b}y + {c} = z")

# Point cloud peak extraction test data
coords = np.transpose(np.nonzero(np.ones((10, 10))))
data = np.ones((100, 1))
data[4] = 2.89
data[10] = 2.578
data[45] = 8.62
data[48] = 4.56
data[58] = 4.12
data[73] = 8.551
data[38] = 11.74
data[12] = 1.61

print(locally_extreme_points(coords, data, threshold_abs=2.0, neighbourhood=2))
