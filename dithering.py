# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 15:12:44 2022

@author: Christian Konstantinov
"""

# TODO: add alpha channel support ?
# TODO: Serpentine diffusion ?

from enum import Enum

import numpy as np
from numba import njit, cfunc
from PIL import Image


class DiffusionMatrix(Enum):
    """Matrices/Kernels for Error Diffusion, the T property holds the transpose of each matrix.

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

    @property
    def T(self):
        xs, ys, coefficients = self.value
        return ys, xs, coefficients


def atkinson(image, bit_depth):
    return dither(image, bit_depth, DiffusionMatrix.ATKINSON.value)


def burkes(image, bit_depth):
    return dither(image, bit_depth, DiffusionMatrix.BURKES.value)


def floyd_steinberg(image, bit_depth):
    return dither(image, bit_depth, DiffusionMatrix.FLOYD_STEINBERG.value)


def false_floyd_steinberg(image, bit_depth):
    return dither(image, bit_depth, DiffusionMatrix.FALSE_FLOYD_STEINBERG.value)


def jarvis_judice_ninke(image, bit_depth):
    return dither(image, bit_depth, DiffusionMatrix.JARVIS_JUDICE_NINKE.value)


def stucki(image, bit_depth):
    return dither(image, bit_depth, DiffusionMatrix.STUCKI.value)


def sierra(image, bit_depth):
    return dither(image, bit_depth, DiffusionMatrix.SIERRA.value)


def two_row_sierra(image, bit_depth):
    return dither(image, bit_depth, DiffusionMatrix.TWO_ROW_SIERRA.value)


def sierra_lite(image, bit_depth):
    return dither(image, bit_depth, DiffusionMatrix.SIERRA_LITE.value)


@cfunc('float64(float64, int64)')
def quantize_uniform(color, palette_size):
    return round(color / 255.0 * palette_size) / palette_size * 255.0


@njit
def dither(image, bit_depth, matrix):
    height, width, channels = image.shape
    palette_size = (1 << bit_depth) - 1
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                color = image[y, x, c]
                quant_color = quantize_uniform(color, palette_size)
                image[y, x, c] = quant_color
                quant_error = color - quant_color
                for u, v, k in zip(*matrix):
                    if 0 <= x + u < width and 0 <= y + v < height:
                        image[y + v, x + u, c] += quant_error * k
    return image
