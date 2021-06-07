"""
THIS REPO IS STILL IN ACTIVE DEVELOPMENT.

Rock extraction and matching for Mars terrains, serving as a reliable compensation for multi-veiw matching.

Rock extraction is currently implemented by pymeanshift, a Python wrapper of Mean Shift algorithm. GPU acceleration may be added in the future.

Copyleft Lang Zhou, zhoulang731@tongji.edu.cn

GitHub: https://github.com/Mars-Rover-Localization/RockMatching

Created May 2021

Last modified June 2021
"""

# Third-party modules
import cv2
import pymeanshift as pms
import numpy as np
import utilities as utl

# Local modules
from config import ROCK_MIN_SIZE

original_image = cv2.imread("sample/Mars_Perseverance_NLF_0102_0676000820_801ECM_N0040372NCAM03101_01_295J.png")
original_image = cv2.GaussianBlur(original_image, (5, 5), 2)
visualized_image = original_image.copy()

(segmented_image, labels_image, number_regions) = pms.segment(original_image, spatial_radius=6, range_radius=4.5,
                                                              min_density=ROCK_MIN_SIZE)
cv2.imwrite("output/seg.png", segmented_image)
cv2.imwrite("output/label.png", labels_image)
print(f"Regions segmented: {number_regions}")

max_size = 0

for i in range(number_regions):
    # Get a boolean image where True pixels belongs to the region
    region_mask = (labels_image == i)
    region_size = np.count_nonzero(region_mask)

    if region_size > max_size:
        max_size = region_size

    region_location = np.nonzero(region_mask)   # Gets non-zero values' locations, in the form of tuple(np.array(xi), np.array(yi))

    edge = utl.edge_extraction(region_mask)
    edge_points = np.transpose(np.nonzero(edge))

    params = utl.ellipse_model_fitting(edge_points)

    fit_result = utl.ellipse_filtering(i, params)

    if not fit_result:
        continue

    print(f"Ellipse fitting succeeded for area {i}")

    point_x = region_location[0][region_size // 2]
    point_y = region_location[1][region_size // 2]
    position = (point_y, point_x)
    visualized_image = cv2.putText(visualized_image, str(region_size), position, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)


print(f"Max region size is {max_size}, defining it as terrain...")

cv2.imshow("vis", visualized_image)
cv2.waitKey()
