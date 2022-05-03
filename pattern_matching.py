import utilities as utl
from YOLOv5_Based_Detection import crater_detection, rock_detection

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


def open3d_icp_wrapper(template: np.ndarray, data: np.ndarray, initial_values: Tuple[float, float, float] = None,
                       max_dist: float = 100, max_iter: int = 2000, vis: bool = True):
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
        initial_guess[0, 3] = initial_values[0]  # x translation
        initial_guess[1, 3] = initial_values[1]  # y translation

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

        ax1.scatter(*template.transpose(), label='Template', s=5, c='blue')
        ax2.scatter(*data.transpose(), label='Data', s=5, c='red', marker='.')
        ax3.scatter(*aligned.transpose(), label='Aligned', s=5, c='green', marker='_')

        ax1.set_title('Template Rocks')
        ax2.set_title('Rocks to Be Aligned')
        ax3.set_title('Aligned Rocks')

        plt.axis('square')

        plt.show()

    return transformation, aligned, reg_p2p


def brute_force_ICP(template: np.ndarray, data: np.ndarray, n: int = 2):
    template, template_scaler = utl.scale_point_cloud(template)
    data, data_scaler = utl.scale_point_cloud(data)

    results, scores = [], []

    for i in range(n):
        for j in range(n):
            transformation, aligned, reg_p2p = open3d_icp_wrapper(template, data, initial_values=(i / n, j / n, 0),
                                                                  max_dist=0.5, vis=False)

            results.append((transformation, aligned, reg_p2p.correspondence_set))
            scores.append(reg_p2p.fitness)

    best_index = np.argmax(scores)
    best_res = results[best_index]

    print(f"Best ICP registration found, fitness: {scores[best_index]} result:\n")
    print(np.asarray(best_res[2]))

    return best_res


if __name__ == '__main__':
    rover_rock_locations_image = np.load('sample/rock_loc_new.npy')

    camera_param = np.array([1.8472689756506643e3, 1.2197869110107422e3, 1.0120676040649414e3, 1.502284])
    camera_param = np.array([1796.98, 1205.67, 981.44, 1802.91, 1259.98, 1040.55, 1502.284])

    res = utl.compute_stereo_3d_coord(rover_rock_locations_image, camera_param)

    print(res)

    rover_rock_locations_ground = res[:, :2]

    plt.scatter(*rover_rock_locations_ground.transpose())

    for index in range(rover_rock_locations_ground.shape[0]):
        plt.text(rover_rock_locations_ground[index, 0], rover_rock_locations_ground[index, 1], str(index))

    plt.show()
    exit()

    uav_rock_locations = rock_detection(r"C:\Users\Lincoln\Project\Moon Field 0306_2\5_Products\Moon Field 0306_2_PlanarMosaic.tif", weight_path='sample/rock_v1.pt', yolov5_path=r"C:\Users\Lincoln\Development\ML\yolov5_rock", image_size=8192).astype(np.float64)

    """
    uav_rock_locations = utl.rescale_uav_points(uav_rock_locations, 8.6e3, 4.49, 1.5e-3, 1.5e-3)

    rover_rock_locations_ground -= rover_rock_locations_ground.mean(axis=0)
    uav_rock_locations -= uav_rock_locations.mean(axis=0)
    """

    uav_rock_locations[:, 1] = 11252 - uav_rock_locations[:, 1]  # Inverse y axis   # 2976

    """
    # open3d_icp_wrapper(template, data, initial_values=(500, 400, 0), vis=True)
    # _, _, icp_res = open3d_icp_wrapper(uav_rock_locations, rover_rock_locations_ground, initial_values=(0, 0, 0.5), max_dist=500, vis=True)

    exit()

    correspondence = np.asarray(icp_res[2])

    rover_matched = rover_rock_locations_ground[correspondence[:, 0].flatten()]
    uav_matched = uav_rock_locations[correspondence[:, 1].flatten()]
    """

    rover_matched = rover_rock_locations_ground[[0, 1, 2, 3, 4, 5, 6]]
    uav_matched = uav_rock_locations[[33, 20, 7, 13, 1, 8, 28]]

    affine_transform = utl.get_affine_transform(rover_matched, uav_matched)
    print(affine_transform)

    rover_reprojected = utl.apply_affine_transform(rover_rock_locations_ground, affine_transform)

    matplotlib.use('TkAgg')

    fig, axs = plt.subplots(1, 2)

    axs[0].scatter(*rover_rock_locations_ground.transpose(), label='Corrected Rover View')
    axs[0].scatter(*rover_matched.transpose(), marker='x', label='Rover View Matched')

    axs[1].scatter(*uav_rock_locations.transpose(), label='UAV View')

    for index in range(uav_rock_locations.shape[0]):
        axs[1].text(uav_rock_locations[index, 0], uav_rock_locations[index, 1], str(index))

    for index in range(rover_rock_locations_ground.shape[0]):
        axs[0].text(rover_rock_locations_ground[index, 0], rover_rock_locations_ground[index, 1], str(index))

    axs[1].scatter(*uav_matched.transpose(), marker='x', label='UAV View Matched')
    axs[1].scatter(*rover_reprojected.transpose(), marker='1', label='Rover View Reprojected')
    # axs[1].invert_yaxis()

    axs[0].legend()
    axs[1].legend()

    plt.show()
