import cv2
import pymeanshift as pms
import numpy as np

original_image = cv2.imread("sample/right_cam.png")
original_image = cv2.GaussianBlur(original_image, (5, 5), 2)

(segmented_image, labels_image, number_regions) = pms.segment(original_image, spatial_radius=6, range_radius=4.5,
                                                              min_density=250)
cv2.imwrite("sample/seg.png", segmented_image)
cv2.imwrite("sample/label.png", labels_image)
print(f"Regions segmented: {number_regions}")

max_size = 0

for i in range(number_regions):
    # Get a boolean image where True pixels belongs to the region
    region_mask = (labels_image == i)
    region_size = np.count_nonzero(region_mask)

    if region_size > max_size:
        max_size = region_size

region = np.nonzero(region_mask)    # Gets non-zero values' locations, in the form of tuple(np.array(xi), np.array(yi))
print(region)

print(f"Max region size is {max_size}, defining it as terrain...")
