# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 15:12:44 2022

@author: Christian Konstantinov
"""

from PIL import Image
import numpy as np

def floyd_steinberg(image, n_bits=3):
    image = image.astype(np.float32)
    cols, rows, _ = image.shape

    u = np.array([1, -1, 0, 1]) # error column indices
    v = np.array([0,  1, 1, 1]) # error row indices
    f_error_bias = np.array([[7, 7, 7,],[3, 3, 3,],[5, 5, 5,],[1, 1, 1,]]) / 16 # error diffusion coefficients
    f_levels = (1 << n_bits) - 1

    for x in range(cols):
        for y in range(rows):
            xn, yn = x + u, y + v
            if np.any(xn < 0) or np.any(xn >= cols) or np.any(yn < 0) or np.any(yn >= rows):
                continue
            quantized_pixel = np.round(image[x, y] / 255 * f_levels) / f_levels * 255
            error = image[x, y] - quantized_pixel
            image[x, y] = quantized_pixel
            image[xn, yn] += error * f_error_bias
    return np.clip(image, 0, 255).astype(np.uint8)

if __name__ == '__main__':
    image = Image.open('images/smile.png')
    input_data = np.array(image)
    output_data = floyd_steinberg(input_data)
    dithered_image = Image.fromarray(output_data)
    dithered_image.save('images/smile_dithered.png')
