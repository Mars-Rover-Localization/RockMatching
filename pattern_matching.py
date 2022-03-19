import utilities as utl
from YOLOv5_Based_Detection import crater_detection

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import open3d as o3d
from cv2 import cv2

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


def open3d_icp_wrapper(template: np.ndarray, data: np.ndarray, initial_values: Tuple[float, float, float] = None, max_dist: float = 100, max_iter: int = 2000, vis: bool = True):
    """
    Wrapper method for Open3D library ICP implementation.

    :param template: Reference points, in form of (N, 2) ndarray
    :param data: Points to be aligned, in form of (M, 2) ndarray
    :param initial_values: (t_X, t_Y, rotation) # TODO: Complete initial guess for rotation
    :param max_dist: Maximum correspondence distance, recommended not to set too small
    :param max_iter: Max iterations performs, too high value is mostly unnecessary
    :param vis: If perform visualization, default as True
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

    if vis:
        matplotlib.use('tkagg')

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 8))
        fig.suptitle("Detection & Matching Result")

        img1 = cv2.imread("test/h.png")
        img2 = cv2.imread("test/l.png")

        ax1.scatter(*template.transpose(), label='Template', s=5, c='blue')
        ax2.scatter(*data.transpose(), label='Data', s=5, c='red', marker='.')
        ax3.scatter(*aligned.transpose(), label='Aligned', s=5, c='green', marker='_')

        ax1.imshow(img1, aspect='equal')
        ax2.imshow(img2, aspect='equal')
        ax3.imshow(img1, aspect='equal')

        ax1.set_title('Template Craters')
        ax2.set_title('Craters to Be Aligned')
        ax3.set_title('Aligned Craters')

        plt.axis('square')
        plt.ylim(img1.shape[0], 0)
        plt.xlim(0, img1.shape[1])

        fig.tight_layout()

        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        plt.savefig('output/vis.png', dpi=400)
        plt.show()

    return transformation, aligned, reg_p2p


def brute_force_ICP(template: np.ndarray, data: np.ndarray, n: int = 2):
    template, template_scaler = utl.scale_point_cloud(template)
    data, data_scaler = utl.scale_point_cloud(data)

    results, scores = [], []

    for i in range(n):
        for j in range(n):
            transformation, aligned, reg_p2p = open3d_icp_wrapper(template, data, initial_values=(i / n, j / n, 0), max_dist=0.5, vis=False)

            results.append((transformation, aligned, reg_p2p.correspondence_set))
            scores.append(reg_p2p.fitness)

    best_index = np.argmax(scores)
    best_res = results[best_index]

    print(f"Best ICP registration found, fitness: {scores[best_index]} result:\n")
    print(best_res[2])

    return best_res


if __name__ == '__main__':
    best_pt_path = r"C:\Users\Lincoln\Development\ML\yolov5_crater\runs\train\exp5\weights\best.pt"
    yolov5_path = r"C:\Users\Lincoln\Development\ML\yolov5_crater"

    with utl.Timer("Detecting Craters..."):
        location, params = crater_detection(["test/h.png", "test/l.png"], weight_path=best_pt_path, yolov5_path=yolov5_path)

    template = expand_points(location[0], params[0])
    data = expand_points(location[1], params[1])

    # open3d_icp_wrapper(template, data, initial_values=(500, 400, 0), vis=True)
    brute_force_ICP(template, data)
