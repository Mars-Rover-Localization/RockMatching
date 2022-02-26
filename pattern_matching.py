import numpy as np
import utilities as utl

import matplotlib.pyplot as plt
import matplotlib
import open3d as o3d
from YOLOv5_Based_Detection import crater_detection

from typing import Tuple
import math

"""
Last edited February 2022: ICP matching implementation is switched to Open3D (http://www.open3d.org/)

Good initial guess is EXTREMELY CRUCIAL for ICP algorithm, please refer to the wrapper method for more info.
"""


def points_on_circumference(center: Tuple[float, float] = (0.0, 0.0), r: float = 50.0, n: int = 100):
    return [
        (
            center[0] + (math.cos(2 * math.pi / n * x) * r),  # x
            center[1] + (math.sin(2 * math.pi / n * x) * r),  # y
        )
        for x in range(0, n + 1)
    ]


def expand_points(location, params):
    assert len(location) == len(params), 'Inconsistent input data'

    res_pts = []

    for index in range(len(location)):
        width, height = params[index]
        avg_radius = (width + height) / 4

        n_points = int(avg_radius * 10)

        res_pts.extend(points_on_circumference(location[index], avg_radius, n_points if n_points > 10 else 10))

    return np.array(res_pts)


def open3d_icp_wrapper(template: np.ndarray, data: np.ndarray, initial_values: Tuple[float, float, float] = None, max_dist: int = 100, max_iter: int = 2000):
    """
    Wrapper method for Open3D library ICP implementation.

    :param template: Reference points, in form of (N, 2) ndarray
    :param data: Points to be aligned, in form of (M, 2) ndarray
    :param initial_values: (t_X, t_Y, rotation) # TODO: Complete initial guess for rotation
    :param max_dist: Maximum correspondence distance, recommended not to set too small
    :param max_iter: Max iterations performs, too high value is mostly unnecessary
    :return: None
    """
    pcd1, pcd2 = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()

    pcd1.points = o3d.utility.Vector3dVector(np.hstack((template, np.zeros((template.shape[0], 1)))))
    pcd2.points = o3d.utility.Vector3dVector(np.hstack((data, np.zeros((data.shape[0], 1)))))

    initial_guess = np.identity(4)

    if initial_values:
        initial_guess[0, 3] = initial_values[0]     # x translation
        initial_guess[1, 3] = initial_values[1]    # y translation

    with utl.Timer('ICP Matching...'):
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd2, pcd1, max_dist, initial_guess,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
        )

    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)

    transformation = reg_p2p.transformation
    aligned = np.array(pcd2.transform(transformation).points)[:, :2]

    matplotlib.use('tkagg')
    plt.scatter(*template.transpose(), label='Template')
    plt.scatter(*data.transpose(), label='Data')
    plt.scatter(*aligned.transpose(), label='Aligned')
    plt.axis('square')
    plt.legend()
    plt.ylim(plt.gca().get_ylim()[::-1])
    plt.show()


if __name__ == '__main__':
    best_pt_path = r"C:\Users\Lincoln\Development\ML\yolov5_crater\runs\train\exp5\weights\best.pt"
    yolov5_path = r"C:\Users\Lincoln\Development\ML\yolov5_crater"

    location, params = crater_detection(["test/h.png", "test/l.png"], weight_path=best_pt_path, yolov5_path=yolov5_path)

    template = expand_points(location[0], params[0])
    data = expand_points(location[1], params[1])

    open3d_icp_wrapper(template, data, initial_values=(500, 400, 0))
