# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 15:12:44 2022

@author: Christian Konstantinov
"""

# TODO: add alpha channel and single color channel support
# TODO: Maybe flatten arrays for use w/ Numba?

from PIL import Image
import numpy as np


def quantize_uniform(image, bit_depth):
    palette_size = (1 << bit_depth) - 1
    return np.round(image / 255 * palette_size) / palette_size * 255


def floyd_steinberg(image, bit_depth=1):
    image = image.astype(np.float32)
    cols, rows, _ = image.shape

    u = np.array([1, -1, 0, 1])  # error column indices
    v = np.array([0,  1, 1, 1])  # error row indices
    diffusion_coefficients = np.array([[7, 7, 7, ], [3, 3, 3, ], [5, 5, 5, ], [
                                      1, 1, 1, ]]) / 16  # error diffusion coefficients

    for x in range(cols):
        for y in range(rows):
            xn, yn = x + u, y + v
            if np.any(xn < 0) or np.any(xn >= cols) or np.any(yn < 0) or np.any(yn >= rows):
                continue
            quantized_pixel = quantize_uniform(image[x, y], bit_depth)
            quant_error = image[x, y] - quantized_pixel
            image[x, y] = quantized_pixel
            image[xn, yn] += quant_error * diffusion_coefficients
    return np.clip(image, 0, 255).astype(np.uint8)


if __name__ == '__main__':
    image = Image.open('images/キャンディーポット.png')
    input_data = np.array(image)
    output_data = floyd_steinberg(input_data)
    dithered_image = Image.fromarray(output_data)
    dithered_image.save('images/キャンディーポット_dithered_1_bit.png')
