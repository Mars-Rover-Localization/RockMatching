"""
This script contains method for crater detection using YOLOv5 framework.
Copyright 2022 Lincoln Zhou, zhoulang731@tongji.edu.cn
"""

import torch
from cv2 import cv2
import numpy as np

import os

from utilities import draw_marker


def crater_detection(images, weight_path: str, yolov5_path: str):
    """
    Detect crater using trained YOLOv5 model file
    :param images: list of image paths (str)
    :param weight_path: YOLOv5 model parameters file's path
    :param yolov5_path: YOLOv5 folder path
    :return: [[[center_x, center_y]]], [[[width, height]]]
    """
    assert len(images) >= 1, 'No valid images'

    try:
        model = torch.hub.load(yolov5_path, 'custom', path=weight_path, source='local')
        model.conf = 0.5
    except:
        print("Model file cannot be loaded, stop.")
        return

    detection_result_loc = []
    detection_result_parm = []

    results = model(images, size=1280).pandas().xyxy

    for index in range(len(images)):
        result = results[index]

        current_image_result_loc = []
        current_image_result_parm = []

        for row in result.itertuples(index=True, name='Pandas'):
            center_x = int((row.xmin + row.xmax) / 2)
            center_y = int((row.ymin + row.ymax) / 2)

            width = row.xmax - row.xmin
            height = row.ymax - row.ymin

            current_image_result_loc.append([center_x, center_y])
            current_image_result_parm.append([width, height])

        detection_result_loc.append(current_image_result_loc)
        detection_result_parm.append(current_image_result_parm)

    return detection_result_loc, detection_result_parm


def rock_detection(image: str, weight_path: str, yolov5_path: str, image_size: int = 1280, conf_thres: float = 0.45, verbose: bool = False) -> np.ndarray:
    """
    Detect crater using trained YOLOv5 model file
    :param image: image path (str)
    :param weight_path: YOLOv5 model parameters file's path
    :param yolov5_path: YOLOv5 folder path
    :param image_size: Image size for YOLO detection, default value suits most close-view photos, for large DOM image please choose a suitable value (preferably 2^n)
    :param conf_thres: Confidence threshold, default as 0.45
    :param verbose: Pass True to allow all display and print methods, default as False
    :return: (N, 2) size np.ndarray
    """

    try:
        model = torch.hub.load(yolov5_path, 'custom', path=weight_path, source='local')
        model.conf = conf_thres
    except:
        print("Model file cannot be loaded, stop.")
        exit()

    detection_result_loc = []

    result = model(image, size=image_size)

    if verbose:
        result.print()
        result.show()

    result = result.pandas().xyxy[0]
    print(result)

    for row in result.itertuples(index=True, name='Pandas'):
        center_x = int((row.xmin + row.xmax) / 2)
        center_y = int((row.ymin + row.ymax) / 2)

        detection_result_loc.append([center_x, center_y])

    return np.array(detection_result_loc)


if __name__ == '__main__':
    rock_locs = rock_detection("sample/102r.png", weight_path='sample/rock_v1.pt', yolov5_path=r"C:\Users\Lincoln\Development\ML\yolov5_rock", image_size=1920)
    # img = draw_marker(cv2.imread('sample/Explorer_HD1080_SN22734452_09-07-19.png'), rock_locs)
    img = cv2.imread("sample/102r.png")

    for loc in rock_locs:
        img = cv2.putText(img, f"{loc[0]}, {loc[1]}", loc, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.85, thickness=1, color=(0, 255, 0))

    cv2.imwrite(r"C:\Users\Lincoln\Desktop\right.png", img)
    exit()

    test_folder_path = r"C:\Users\Lincoln\Development\ML\yolov5_crater\test\images"
    best_pt_path = r"C:\Users\Lincoln\Development\ML\yolov5_crater\runs\train\exp5\weights\best.pt"
    yolov5_path = r"C:\Users\Lincoln\Development\ML\yolov5_crater"
    test_images = os.listdir(test_folder_path)

    print(crater_detection([os.path.join(test_folder_path, x) for x in test_images], weight_path=best_pt_path,
                           yolov5_path=yolov5_path))
