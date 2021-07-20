"""
This file implements the basic idea of extracting rocks from point cloud.

Since no field test has been conducted due to the lack of stereo camera, the code currently only serve as demonstration.

Copyleft Lang Zhou, zhoulang731@tongji.edu.cn

GitHub: https://github.com/Mars-Rover-Localization/RockMatching

Created May 2021

Last modified July 2021
"""

# Third-party modules
import numpy as np

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
    n = points.shape[0]
    A = np.hstack((points[:, 0:2], np.ones((n, 1))))
    B = points[:, 2]

    x = np.linalg.inv((A.transpose() @ A)) @ A.transpose() @ B
    a, b, c = x.transpose().tolist()

    return a, b, c


# Test data
x = np.array([[0.274791784, -1.001679346, -1.851320839, 0.365840754]])
y = np.array([[-1.155674199, -1.215133985, 0.053119249, 1.162878076]])
z = np.array([[1.216239624, 0.764265677, 0.956099579, 1.198231236]])
test_points = np.hstack((x.transpose(), y.transpose(), z.transpose()))

with utl.Timer("Plane fitting..."):
    a, b, c = plane_fitting(test_points)

print(f"Fitted plane: {a}x + {b}y + {c} = z")
