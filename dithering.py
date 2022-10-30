# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 15:12:44 2022

@author: Christian Konstantinov
"""

# TODO: add alpha channel support ?

from PIL import Image
import numpy as np
from numba import njit, prange


def quantize_uniform(image, bit_depth):
    palette_size = (1 << bit_depth) - 1
    return np.round(image / 255. * palette_size) / palette_size * 255.


@njit
def floyd_steinberg(image, bit_depth):
    height, width, channels = image.shape
    xs = (1, -1, 0, 1)
    ys = (0,  1, 1, 1)
    kernel = (7 / 16, 3 / 16, 5 / 16, 1 / 16)
    palette_size = (1 << bit_depth) - 1

    for y in prange(height):
        for x in range(width):
            for c in range(channels):
                color = image[y, x, c]
                quant_color = round(
                    color / 255. * palette_size) / palette_size * 255.
                image[y, x, c] = quant_color
                quant_error = color - quant_color
                for u, v, k in zip(xs, ys, kernel):
                    if 0 <= x + u < width and 0 <= y + v < height:
                        image[y + v, x + u, c] += quant_error * k
    return image


def verify(image, bit_depth, num_channels=3) -> bool:
    palette_size = 1
    for i in range(num_channels):
        palette_size *= np.unique(image[:, :, i]).shape[0]
    expected_palette_size = (1 << bit_depth) ** num_channels
    return palette_size <= expected_palette_size


if __name__ == '__main__':
    image = Image.open('images/hatsune_miku.png')
    input_data = np.array(image)
    bit_depth = 2
    output_data = floyd_steinberg(input_data.astype(
        np.float32), bit_depth).astype(np.uint8)
    dithered_image = Image.fromarray(output_data)
    assert verify(output_data, bit_depth)
    dithered_image.save('images/hatsune_miku_dithered_2_bits.png')
