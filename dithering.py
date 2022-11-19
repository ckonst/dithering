# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 15:12:44 2022

@author: Christian Konstantinov
"""

# TODO: Add alpha channel support
# TODO: Add serpentine diffusion
# TODO: Add Gamma Correction

from functools import lru_cache
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
        (1, 0, 0.125),
        (2, 0, 0.125),
        (-1, 1, 0.125),
        (0, 1, 0.125),
        (1, 1, 0.125),
        (0, 2, 0.125)
    )
    BURKES = (
        (1, 0, 0.25),
        (2, 0, 0.125),
        (-2, 1, 0.0625),
        (-1, 1, 0.125),
        (0, 1, 0.25),
        (1, 1, 0.125),
        (2, 1, 0.0625)
    )
    FLOYD_STEINBERG = (
        (1, 0, 0.4375),
        (-1, 1, 0.1875),
        (0, 1, 0.3125),
        (1, 1, 0.0625)
    )
    FALSE_FLOYD_STEINBERG = (
        (1, 0, 0.375),
        (0, 1, 0.375),
        (1, 1, 0.25)
    )
    JARVIS_JUDICE_NINKE = (
        (1, 0, 0.14583333333333334),
        (2, 0, 0.10416666666666667),
        (-2, 1, 0.0625),
        (-1, 1, 0.10416666666666667),
        (0, 1, 0.14583333333333334),
        (1, 1, 0.10416666666666667),
        (2, 1, 0.0625),
        (-2, 2, 0.020833333333333332),
        (-1, 2, 0.0625),
        (0, 2, 0.10416666666666667),
        (1, 2, 0.0625),
        (2, 2, 0.020833333333333332)
    )
    STUCKI = (
        (1, 0, 0.16666666666666666),
        (2, 0, 0.08333333333333333),
        (-2, 1, 0.041666666666666664),
        (-1, 1, 0.08333333333333333),
        (0, 1, 0.16666666666666666),
        (1, 1, 0.08333333333333333),
        (2, 1, 0.041666666666666664),
        (-2, 2, 0.020833333333333332),
        (-1, 2, 0.041666666666666664),
        (0, 2, 0.08333333333333333),
        (1, 2, 0.041666666666666664),
        (2, 2, 0.020833333333333332)
    )
    SIERRA = (
        (1, 0, 0.15625),
        (2, 0, 0.09375),
        (-2, 1, 0.0625),
        (-1, 1, 0.125),
        (0, 1, 0.15625),
        (1, 1, 0.125),
        (2, 1, 0.0625),
        (-1, 2, 0.0625),
        (0, 2, 0.09375),
        (1, 2, 0.0625)
    )
    TWO_ROW_SIERRA = (
        (1, 0, 0.25),
        (2, 0, 0.1875),
        (-2, 1, 0.0625),
        (-1, 1, 0.125),
        (0, 1, 0.1875),
        (1, 1, 0.125),
        (2, 1, 0.0625)
    )
    SIERRA_LITE = (
        (1, 0, 0.5),
        (-1, 1, 0.25),
        (0, 1, 0.25)
    )

    @property
    @lru_cache(maxsize=1)
    def hysteresis(self):
        """
        Return the matrix rotated 180 degrees for hysteresis filter preturbation.
        Calculate the matrix once retrieve from the cache on subsequent calls.
        """
        return tuple((-i, -j, c) for i, j, c in self.value)


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
def quantize_uniform(color: float, palette_size: float) -> int:
    return round(color / 255.0 * palette_size) / palette_size * 255.0


@njit
def dither(image: np.ndarray, bit_depth: int, matrix: DiffusionMatrix) -> np.ndarray:
    height, width, channels = image.shape
    palette_size = (1 << bit_depth) - 1
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                color = image[y, x, c]
                quant_color = quantize_uniform(color, palette_size)
                image[y, x, c] = quant_color
                quant_error = color - quant_color
                for u, v, k in matrix:
                    if 0 <= x + u < width and 0 <= y + v < height:
                        image[y + v, x + u, c] += quant_error * k
    return image
