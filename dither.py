# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 15:12:44 2022

@author: Christian Konstantinov
"""

from PIL import Image
import numpy as np

def floyd_steinberg(image, n_bits=2):
    image = image.astype(np.float32)
    cols, rows, _ = image.shape
    f_error_bias = np.array([[7, 3, 5, 1],
                             [7, 3, 5, 1],
                             [7, 3, 5, 1]], dtype=np.float32).T / 16
    u = np.array([1, -1, 0, 1])
    v = np.array([0, 1, 1, 1])
    f_levels = (1 << n_bits) - 1

    for x in range(cols):
        for y in range(rows):
            quantized_pixel = np.round(image[x, y] / 255. * f_levels) / f_levels * 255.
            error = image[x, y].astype(np.float32) - quantized_pixel.astype(np.float32)
            image[x, y] = quantized_pixel
            image[np.clip(x + u, 0, cols-1), np.clip(y + v, 0, rows-1)] += np.clip(error * f_error_bias, 0, 255)
    return np.clip(image, 0, 255).astype(np.uint8)

if __name__ == '__main__':
    image = Image.open('images/hatsune_miku.png')
    input_data = np.array(image)
    output_data = floyd_steinberg(input_data)
    dithered_image = Image.fromarray(output_data)
    dithered_image.save('images/miku_dithered.png')
