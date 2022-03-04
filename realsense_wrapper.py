import pyrealsense2 as rs
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
import open3d as o3d


point_cloud = rs.pointcloud()

pipe = rs.pipeline()

pipe.start()

while True:
    frames = pipe.wait_for_frames()
    depth = frames.get_depth_frame()
    color = frames.get_color_frame()

    # RGB data is unnecessary
    # point_cloud.map_to(mapped=color)
    points = point_cloud.calculate(depth)

    vtx = np.asanyarray(points.get_vertices())
    vtx = structured_to_unstructured(vtx)

    print(vtx)
    print(vtx.shape)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vtx)
    o3d.visualization.draw_geometries([pcd])
