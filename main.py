import cv2
import pymeanshift as pms
import numpy as np

original_image = cv2.imread("sample/1.png")
original_image = cv2.GaussianBlur(original_image, (5, 5), 2)

(segmented_image, labels_image, number_regions) = pms.segment(original_image, spatial_radius=6, range_radius=4.5,
                                                              min_density=250)
cv2.imwrite("sample/seg.png", segmented_image)
cv2.imwrite("sample/label.png", labels_image)
print(number_regions)

max_count = 0
for i in range(number_regions):
    # Get a boolean image where True pixels belongs to the region
    region_mask = (labels_image == i)
    region_area = np.count_nonzero(region_mask)
    if region_area > max_count:
        max_count = region_area
