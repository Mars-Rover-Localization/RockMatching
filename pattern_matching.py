import numpy as np


def find_corner_point(area, identifier):
    """
    :param area: numpy.ndarray, area containing interest points
    :param identifier: int, specific value distinguishing interest points from other pixels
    :return: numpy.ndarray, corner point's position, in format of [x, y]
    """
    mask = (area == identifier)
    coordinates = np.nonzero(mask)

    distance = [coordinates[0][index] ** 2 + coordinates[1][index] ** 2 for index in range(len(coordinates[0]))]
    min_distance_index = distance.index(min(distance))

    return np.transpose(coordinates)[min_distance_index]
