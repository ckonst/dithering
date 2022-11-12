# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 15:12:44 2022

@author: Christian Konstantinov
"""

# TODO: add alpha channel support ?
# TODO: Serpentine diffusion ?

from functools import wraps
from enum import Enum
from typing import Tuple

import numpy as np
from numba import njit, prange
from PIL import Image


class DiffusionMatrix(Enum):
    """Matrices/Kernels for Error Diffusion

    Args:
        Enum (Tuple[Tuple[Number]]): Tuple of (x indices, y indices, coefficients)
    """

    ATKINSON = (
        (1, 2, -1, 0, 1, 0),
        (0, 0,  1, 1, 1, 2),
        (1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8)
    )
    BURKES = (
        (1, 2, -2, -1, 0, 1, 2),
        (0, 0,  1,  1, 1, 1, 1),
        (8 / 32, 4 / 32, 2 / 32, 4 / 32, 8 / 32, 4 / 32, 2 / 32)
    )
    FLOYD_STEINBERG = (
        (1, -1, 0, 1),
        (0,  1, 1, 1),
        (7 / 16, 3 / 16, 5 / 16, 1 / 16)
    )
    FALSE_FLOYD_STEINBERG = (
        (1, 0, 1),
        (0, 1, 1),
        (3 / 8, 3 / 8, 2 / 8)
    )
    JARVIS_JUDICE_NINKE = (
        (1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2),
        (0, 0,  1,  1, 1, 1, 1,  2,  2, 2, 2, 2),
        (7 / 48, 5 / 48, 3 / 48, 5 / 48, 7 / 48, 5 / 48,
         3 / 48, 1 / 48, 3 / 48, 5 / 48, 3 / 48, 1 / 48)
    )
    STUCKI = (
        (1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2),
        (0, 0,  1,  1, 1, 1, 1,  2,  2, 2, 2, 2),
        (8 / 48, 4 / 48, 2 / 48, 4 / 48, 8 / 48, 4 / 48,
         2 / 48, 1 / 48, 2 / 48, 4 / 48, 2 / 48, 1 / 48)
    )
    SIERRA = (
        (1, 2, -2, -1, 0, 1, 2, -1, 0, 1),
        (0, 0,  1,  1, 1, 1, 1,  2, 2, 2),
        (5 / 32, 3 / 32, 2 / 32, 4 / 32, 5 / 32,
         4 / 32, 2 / 32, 2 / 32, 3 / 32, 2 / 32)
    )
    TWO_ROW_SIERRA = (
        (1, 2, -2, -1, 0, 1, 2),
        (0, 0,  1,  1, 1, 1, 1),
        (4 / 16, 3 / 16, 1 / 16, 2 / 16, 3 / 16, 2 / 16, 1 / 16)
    )
    SIERRA_LITE = (
        (1, -1, 0),
        (0,  1,  1),
        (2 / 4, 1 / 4, 1 / 4)
    )


def transpose(xs, ys, matrix):
    return ys, xs, matrix


# Set a field that points to the Transpose Matrices
setattr(
    DiffusionMatrix, 'T', Enum('TransposeDiffusionMatrix', {
        m.name: transpose(*m.value) for m in DiffusionMatrix
    })
)


def _quantize_uniform(image, bit_depth):
    palette_size = (1 << bit_depth) - 1
    return np.round(image / 255. * palette_size) / palette_size * 255.


def floyd_steinberg(image, bit_depth):
    return dither(image, bit_depth, *DiffusionMatrix.FLOYD_STEINBERG.value)


@ njit
def dither(image, bit_depth, matrix):
    height, width, channels = image.shape
    palette_size = (1 << bit_depth) - 1
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                color = image[y, x, c]
                quant_color = round(
                    color / 255. * palette_size) / palette_size * 255.
                image[y, x, c] = quant_color
                quant_error = color - quant_color
                for u, v, k in zip(*matrix):
                    if 0 <= x + u < width and 0 <= y + v < height:
                        image[y + v, x + u, c] += quant_error * k
    return image


# TODO: move to unit test
def _verify(image, bit_depth, num_channels=3) -> bool:
    palette_size = 1
    for i in range(num_channels):
        palette_size *= np.unique(image[:, :, i]).shape[0]
    expected_palette_size = (1 << bit_depth) ** num_channels
    return palette_size <= expected_palette_size


# TODO: Move to another file
if __name__ == '__main__':
    input_path = 'images/hatsune_miku.png'
    output_path = 'images/hatsune_miku_dithered_2_bits.png'
    image = Image.open(input_path)
    print(f'Loaded image with {input_path = }')
    input_data = np.array(image).astype(np.float32)
    bit_depth = 2
    output_data = floyd_steinberg(input_data, bit_depth).astype(np.uint8)
    dithered_image = Image.fromarray(output_data)
    assert _verify(output_data, bit_depth)
    dithered_image.save(output_path)
    print(f'Saved image with {output_path = }')
