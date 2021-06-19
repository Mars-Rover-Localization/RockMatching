"""
This module contains global settings for Rock Matching project.

Please notice that the numeric values defined here should be adjusted according to testing environment.

For example, minimum size of rocks should be increased if input image has a high resolution.

Copyleft Lang Zhou, zhoulang731@tongji.edu.cn

GitHub: https://github.com/Mars-Rover-Localization/RockMatching

Created May 2021

Last modified June 2021
"""


ROCK_MIN_SIZE = 50  # Minimum rock size

# Suppose the image is a m*n matrix, the program will only analyze part of the whole image since distortion will be
# high in upper area of the image.
# In our implementation, extracted rocks whose rows falling within [0, m*AREA_UPPER_BOUND_RATIO] will be ignored.
# We suggest setting the ratio around 0.15
AREA_UPPER_BOUND_RATIO = 0.15
