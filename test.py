import itertools
import numpy as np
import matplotlib.pyplot as plt

import utilities as utl
from YOLOv5_Based_Detection import crater_detection, rock_detection

import matplotlib
import open3d as o3d
from probreg import cpd, bcpd
from cv2 import cv2

from typing import Tuple
import math
import copy


def generate_cpp_code(points: np.ndarray, start_index=0):
    for index in range(points.shape[0]):
        pt = points[index]

        num = index + start_index

        l1 = f"Vector2d ul{num}({pt[0]}, {pt[1]}), ur{num}({pt[2]}, {pt[3]});"
        l2 = f"Vector3d p{num} = sm.intersect(ul{num}, sm.cam_l, ur{num}, sm.cam_r);"
        l3 = f"cout<<p{num}<<endl;"

        print(f"{l1}\n{l2}\n{l3}\n")


def read_output():
    with open('output.txt') as file:
        nums = file.readlines()

    temp = []

    for index in range(0, len(nums) - 2, 3):
        temp.append([float(nums[index].strip()), float(nums[index + 1].strip())])

    temp = np.array(temp)

    plt.scatter(*temp.transpose())

    for index in range(temp.shape[0]):
        plt.text(temp[index, 0], temp[index, 1], str(index))

    plt.show()

    return temp


# temp = np.load('sample/rock_ori.npy')
# generate_cpp_code(temp, 1)

rover_rock_locations_image = np.load('sample/rock_loc_new.npy')

rover_rock_locations_ground = read_output()

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

rover_matched = rover_rock_locations_ground[[1, 2, 4, 7]]   # [[1, 2, 3, 4, 5, 6, 7]]
uav_matched = uav_rock_locations[[20, 7, 1, 16]]    # [[20, 7, 13, 1, 8, 28, 16]]

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
