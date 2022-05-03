"""
THIS REPO IS STILL IN ACTIVE DEVELOPMENT.

This script contains some methods used in main function

Copyleft Lang Zhou, zhoulang731@tongji.edu.cn

GitHub: https://github.com/Mars-Rover-Localization/RockMatching

Created May 2021

Last modified April 2022
"""

# Built-in modules
import math
from contextlib import contextmanager
import time

# Third-party modules
from scipy import ndimage
from scipy.interpolate import griddata
from sklearn.preprocessing import MinMaxScaler
from skimage.segmentation import slic, mark_boundaries
from skimage.measure import EllipseModel
from skimage.future import graph
import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt
import open3d as o3d
import seaborn as sns
from tqdm import tqdm

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

    if ellipse_area < ROCK_MIN_SIZE:  # Temporarily remove maximum size detection
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
                    # new_size = ellipse_x[0] + ellipse_y[0]
                    continue  # Too extreme?
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


def interpolate_missing_pixels(image: np.ndarray, mask: np.ndarray, method: str = 'nearest', fill_value: int = 0):
    """
    :param image: a 2D image
    :param mask: a 2D boolean image, True indicates missing values
    :param method: interpolation method, one of
        'nearest', 'linear', 'cubic'.
    :param fill_value: which value to use for filling up data outside the
        convex hull of known pixel values.
        Default is 0, Has no effect for 'nearest'.
    :return: the image with missing values interpolated
    """

    h, w = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    known_x = xx[~mask]
    known_y = yy[~mask]
    known_v = image[~mask]
    missing_x = xx[mask]
    missing_y = yy[mask]

    interp_values = griddata(
        (known_x, known_y), known_v, (missing_x, missing_y),
        method=method, fill_value=fill_value
    )

    interp_image = image.copy()
    interp_image[missing_y, missing_x] = interp_values

    return interp_image


def point_cloud_to_DEM(points: np.ndarray, grid_size: int = 500, interpolation: str = 'linear',
                       save_path: str = None) -> np.ndarray:
    """
    Generate DEM grid from point cloud data.

    :param points: 3D points in (N, 3) shape
    :param grid_size: The desired resolution (grid_size * grid_size) of generated DEM
    :param interpolation: Optional, whether to perform interpolation to fill empty 'holes' inside DEM data. Pass None to disable this option.
    :param save_path: Optional. String path to a .npy file if generated DEM needs to be saved locally.

    :return: DEM array in shape (grid_size, grid_size)
    """

    assert interpolation in ['nearest', 'linear', 'cubic'], 'Invalid interpolation option'

    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])

    x_interval = (x_max - x_min) / grid_size
    y_interval = (y_max - y_min) / grid_size

    grid = np.zeros((grid_size, grid_size))

    print("Generating grid...")

    for (grid_x, grid_y) in tqdm(np.ndindex((grid_size, grid_size))):
        temp = points[
            (x_min + grid_x * x_interval < points[:, 0]) & (points[:, 0] < x_min + (grid_x + 1) * x_interval) & (
                    y_min + grid_y * y_interval < points[:, 1]) & (
                    points[:, 1] < y_min + (grid_y + 1) * y_interval)]

        if temp.size != 0:
            grid[grid_x, grid_y] = np.mean(temp, axis=0)[2]

    if interpolation:
        empty_val_mask = grid == 0
        grid = interpolate_missing_pixels(grid, empty_val_mask, interpolation)

    if save_path:
        np.save(save_path, grid)

    sns.color_palette("Spectral", as_cmap=True)
    sns.heatmap(grid, mask=(grid == 0), cmap='Spectral')
    plt.show()

    return grid


def DEM_surface_fitting(grid: np.ndarray, method: str = 'poly'):
    assert method in ['poly', 'linear'], 'Invalid fitting method'

    h, w = grid.shape

    x, y = np.mgrid[:h, :w]

    x, y = x.reshape((-1, 1)), y.reshape((-1, 1))

    if method == 'poly':
        A = np.hstack((x ** 2, y ** 2, x * y, x, y, np.ones((h * w, 1))))
    else:
        A = np.hstack((x, y, np.ones((h * w, 1))))

    params = (np.linalg.inv(A.transpose() @ A) @ A.transpose() @ grid.reshape((-1, 1))).flatten().tolist()

    return params


def DEM_find_local_outlier(grid: np.ndarray, surface_equation: list[float, float, float], epsilon: float = 2):
    assert len(surface_equation) in [3, 6], 'Invalid parameter'

    h, w = grid.shape
    x, y = np.mgrid[:h, :w]

    if len(surface_equation) == 6:
        a, b, c, d, e, f = surface_equation
        fitted_values = a * (x ** 2) + b * (y ** 2) + c * x * y + d * x + e * y + f
    else:
        a, b, c = surface_equation
        fitted_values = a * x + b * y + c

    diff = grid - fitted_values

    mean, std = np.mean(diff), np.std(diff)

    maxima = np.array(np.where(diff > mean + epsilon * std))
    minima = np.array(np.where(diff < mean - epsilon * std))

    return maxima, minima


def DEM_find_global_outlier(dem: np.ndarray, split_resolution: int = 10):
    h, w = dem.shape

    h_step, w_step = h // split_resolution, w // split_resolution
    window_area = h_step * w_step

    max_loc, min_loc = [], []

    for i, j in np.ndindex((split_resolution, split_resolution)):
        window = dem[h_step * i: h_step * (i + 1), w_step * j: w_step * (j + 1)]

        if np.count_nonzero(window) != window_area:
            continue

        window_surface_eq = DEM_surface_fitting(window, method='poly')

        window_max, window_min = DEM_find_local_outlier(window, window_surface_eq, 5)

        window_max[0, :] += h_step * i
        window_max[1, :] += w_step * j

        window_min[0, :] += h_step * i
        window_min[1, :] += w_step * j

        max_loc.append(window_max)
        min_loc.append(window_min)

    max_loc = np.hstack(max_loc)
    min_loc = np.hstack(min_loc)

    return max_loc, min_loc


def draw_marker(img: np.ndarray, locations: np.ndarray):
    assert locations.shape[1] == 2 and len(locations.shape) == 2, 'Invalid location data'

    for location in locations:
        img = cv2.drawMarker(img, position=location, color=(0, 255, 0), markerType=cv2.MARKER_TRIANGLE_UP, markerSize=20, thickness=5)

    return img


def scale_point_cloud(points: np.ndarray):
    assert points.shape[1] == 2, 'Invalid input data shape'

    scaler = MinMaxScaler()

    return scaler.fit_transform(points), scaler


def compute_stereo_3d_coord(points: np.ndarray, camera_param: np.ndarray) -> np.ndarray:
    """
    Compute 3D coordinates from corresponding points on a RECTIFIED stereo pair.

    :param points: (N, 4) array of left and right image pixel coordinate, in order of (x, y, x, y)
    :param camera_param: (4, ) array of (f, cx, cy, stereo_baseline)
    :return: (N, 3) array of 3D coordinates
    """

    # assert len(camera_param) == 4, 'Invalid camera parameters'
    assert points.shape[1] == 4, 'Invalid points data shape'

    # f, cx, cy, baseline = camera_param.flatten()
    l_f, l_cx, l_cy, r_f, r_cx, r_cy, baseline = camera_param.flatten()

    results = []

    for index in range(points.shape[0]):
        x1, y1, x2, y2 = points[index]

        x1 -= l_cx
        x2 -= r_cx

        y1 = l_cy - y1
        y2 = r_cy - y2  # We assume y1 = y2 since the stereo pair is rectified

        disparity = x1 - x2

        X = baseline * x1 / disparity

        Z = baseline * l_f / disparity

        Y = 0.5 * baseline * (y1 + y2) / disparity + 0.2 * Z

        results.append([X, Y, Z])

    return np.array(results)


def rescale_uav_points(points: np.ndarray, height: float, f: float, pixel_size_x, pixel_size_y):
    for index in range(points.shape[0]):
        x, y = points[index]

        points[index, 0] = height * x * pixel_size_x / f
        points[index, 1] = height * y * pixel_size_y / f

    return points


def get_affine_transform(src: np.ndarray, dst: np.ndarray):
    assert src.shape == dst.shape, 'Invalid input shape'

    A = np.hstack((src, np.ones((src.shape[0], 1))))
    B1, B2 = dst[:, 0], dst[:, 1]

    X_t = np.linalg.lstsq(A, B1, rcond=None)[0]
    Y_t = np.linalg.lstsq(A, B2, rcond=None)[0]

    return np.vstack((X_t.transpose(), Y_t.transpose()))


def apply_affine_transform(src: np.ndarray, transform: np.ndarray):
    a, b, c, d, e, f = transform.flatten()

    res = []

    for pt in src:
        x, y = pt.flatten()

        t_x = a * x + b * y + c
        t_y = d * x + e * y + f

        res.append([t_x, t_y])

    return np.array(res)


if __name__ == '__main__':
    # Test only

    data = np.load('sample/rock_loc.npy')

    camera_param = np.array([1059.19, 994.56, 549.37, 120])

    res = compute_stereo_3d_coord(data, camera_param)

    print(res)

    plt.scatter(*res[:, :2].transpose())

    for index in range(res.shape[0]):
        plt.text(res[index, 0], res[index, 1], f"{res[index]}")

    plt.show()


