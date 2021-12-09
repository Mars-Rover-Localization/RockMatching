"""
THIS REPO IS STILL IN ACTIVE DEVELOPMENT.

Rock extraction and matching for Mars terrains, serving as a reliable compensation for multi-veiw matching.

Rock extraction is currently implemented using segmentation algorithm (currently SLIC and MeanShift).

Copyleft Lang Zhou, zhoulang731@tongji.edu.cn

GitHub: https://github.com/Mars-Rover-Localization/RockMatching

Created May 2021

Last modified December 2021
"""

# Third-party modules
from cv2 import cv2
import pymeanshift as pms
import numpy as np

# Local modules
from config import ROCK_MIN_SIZE
import utilities as utl


def rock_extraction(image: str, method: str):
    """
    Extract rocks using segmentation methods.
    :param image: Image path
    :param method: Segmentation algorithm used, ['meanshift' | 'slic']
    :return: None
    """
    assert method in ['meanshift', 'slic'], 'No matching algorithm'

    # TODO: Using 3D point cloud reconstructed from stereo camera to extract rocks more precisely
    original_image = cv2.imread(image)
    visualized_image = original_image.copy()

    if method == 'meanshift':
        original_image = cv2.GaussianBlur(original_image, (5, 5), 2)

        with utl.Timer("Mean shift segmenting..."):
            segmented_image, labels_image, number_regions = pms.segment(original_image, spatial_radius=6,
                                                                        range_radius=4.5,
                                                                        min_density=ROCK_MIN_SIZE)
    elif method == 'slic':
        with utl.Timer("SLIC segmenting..."):
            segmented_image, labels_image, number_regions = utl.slic_wrapper(original_image)

    cv2.imwrite("output/seg.png", segmented_image)
    utl.save_labels("output/label.png", labels_image)
    print(f"Regions segmented: {number_regions}")

    max_size = 0

    extracted_rocks = []

    with utl.Timer("Shape analyzing..."):
        for i in range(number_regions):
            # Get a boolean image where True pixels belongs to the region
            region_mask = (labels_image == i)
            region_size = np.count_nonzero(region_mask)

            if region_size > max_size:
                max_size = region_size

            region_location = np.nonzero(
                region_mask)  # Gets non-zero values' locations, in the form of tuple(np.array(xi), np.array(yi))

            edge = utl.edge_extraction(region_mask)
            edge_points = np.transpose(np.nonzero(edge))

            params = utl.ellipse_model_fitting(edge_points)     # params: xc, yc, a, b, c, e, theta

            fit_result = utl.ellipse_filtering(i, params)       # fit_result is the same as params if succeeded

            if not fit_result:
                continue

            print(f"Ellipse fitting succeeded for area {i}")  # Print method here is for inspection use, will be deprecated in release

            extracted_rocks.append([region_size, list(fit_result), edge])

            visualized_image[tuple(edge_points.transpose())] = [0, 0, 255]

            point_x = region_location[0][region_size // 2]
            point_y = region_location[1][region_size // 2]
            position = (point_y, point_x)
            visualized_image = cv2.putText(visualized_image, str(region_size), position, cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                           (255, 0, 0), 1)

        sparsed_rocks = utl.ellipse_sparsing(extracted_rocks)

        print(len(extracted_rocks), len(sparsed_rocks))

        sparsed_vs = utl.visualize_rocks(original_image, sparsed_rocks)
        cv2.imshow("sparsed", sparsed_vs)

    print(f"Max region size is {max_size}, defining it as terrain...")

    cv2.imwrite("output/visualized.png", visualized_image)
    cv2.imshow("vis", visualized_image)
    cv2.waitKey()

    return  # TODO: Return numpy.ndarray like descriptor for all rocks extracted


if __name__ == '__main__':
    print(__doc__)

    rock_extraction("sample/Mars_Perseverance_NLF_0102_0676000820_801ECM_N0040372NCAM03101_01_295J.png", 'slic')
