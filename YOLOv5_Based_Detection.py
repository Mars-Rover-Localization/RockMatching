"""
This script contains method for crater detection using YOLOv5 framework.
Copyright 2022 Lincoln Zhou, zhoulang731@tongji.edu.cn
"""


import torch

import os


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


if __name__ == '__main__':
    test_folder_path = r"C:\Users\Lincoln\Development\ML\yolov5_crater\test\images"
    best_pt_path = r"C:\Users\Lincoln\Development\ML\yolov5_crater\runs\train\exp5\weights\best.pt"
    yolov5_path = r"C:\Users\Lincoln\Development\ML\yolov5_crater"
    test_images = os.listdir(test_folder_path)

    print(crater_detection([os.path.join(test_folder_path, x) for x in test_images], weight_path=best_pt_path, yolov5_path=yolov5_path))
