# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 15:12:44 2022

@author: Christian Konstantinov
"""

# TODO: Add alpha channel support
# TODO: Add serpentine diffusion
# TODO: Add Gamma Correction
# TODO: Add filter preturbation

from functools import lru_cache
from enum import Enum
from typing import Tuple

import numpy as np
from numba import njit, cfunc
from PIL import Image


class DiffusionMatrix(Enum):
    """Matrices/Kernels for Error Diffusion, the T property holds the transpose of each matrix.

    Args:
        Enum (Tuple[Tuple[Number]]): Tuple of (x indices, y indices, coefficients)
    """

    ATKINSON = (
        (1, 0, 1 / 8),
        (2, 0, 1 / 8),
        (-1, 1, 1 / 8),
        (0, 1, 1 / 8),
        (1, 1, 1 / 8),
        (0, 2, 1 / 8)
    )
    BURKES = (
        (1, 0, 8 / 32),
        (2, 0, 4 / 32),
        (-2, 1, 2 / 32),
        (-1, 1, 4 / 32),
        (0, 1, 8 / 32),
        (1, 1, 4 / 32),
        (2, 1, 2 / 32)
    )
    FLOYD_STEINBERG = (
        (1, 0, 7 / 16),
        (-1, 1, 3 / 16),
        (0, 1, 5 / 16),
        (1, 1, 1 / 16)
    )
    FALSE_FLOYD_STEINBERG = (
        (1, 0, 3 / 8),
        (0, 1, 3 / 8),
        (1, 1, 2 / 8)
    )
    JARVIS_JUDICE_NINKE = (
        (1, 0, 7 / 48),
        (2, 0, 5 / 48),
        (-2, 1, 3 / 48),
        (-1, 1, 5 / 48),
        (0, 1, 7 / 48),
        (1, 1, 5 / 48),
        (2, 1, 3 / 48),
        (-2, 2, 1 / 48),
        (-1, 2, 3 / 48),
        (0, 2, 5 / 48),
        (1, 2, 3 / 48),
        (2, 2, 1 / 48)
    )
    STUCKI = (
        (1, 0, 8 / 48),
        (2, 0, 4 / 48),
        (-2, 1, 2 / 48),
        (-1, 1, 4 / 48),
        (0, 1, 8 / 48),
        (1, 1, 4 / 48),
        (2, 1, 2 / 48),
        (-2, 2, 1 / 48),
        (-1, 2, 2 / 48),
        (0, 2, 4 / 48),
        (1, 2, 2 / 48),
        (2, 2, 1 / 48)
    )
    SIERRA = (
        (1, 0, 5 / 32),
        (2, 0, 3 / 32),
        (-2, 1, 2 / 32),
        (-1, 1, 4 / 32),
        (0, 1, 5 / 32),
        (1, 1, 4 / 32),
        (2, 1, 2 / 32),
        (-1, 2, 2 / 32),
        (0, 2, 3 / 32),
        (1, 2, 2 / 32)
    )
    TWO_ROW_SIERRA = (
        (1, 0, 4 / 16),
        (2, 0, 3 / 16),
        (-2, 1, 1 / 16),
        (-1, 1, 2 / 16),
        (0, 1, 3 / 16),
        (1, 1, 2 / 16),
        (2, 1, 1 / 16)
    )
    SIERRA_LITE = (
        (1, 0, 2 / 4),
        (-1, 1, 1 / 4),
        (0, 1, 1 / 4)
    )

    @property
    @lru_cache(maxsize=1)
    def hysteresis(self):
        """
        Return the matrix rotated 180 degrees for hysteresis filter preturbation.
        Calculate the matrix once retrieve from the cache on subsequent calls.
        """
        return tuple((-x, -y, c) for x, y, c in self.value)

    @property
    @lru_cache(maxsize=1)
    def hysteresis_reversed(self):
        """
        Return the matrix rotated 180 degrees for hysteresis filter preturbation.
        Calculate the matrix once retrieve from the cache on subsequent calls.
        """
        return tuple((x, -y, c) for x, y, c in self.value)

    @property
    @lru_cache(maxsize=1)
    def reversed(self):
        """
        Return the matrix flipped horizontally for odd-row serpentine diffusion.
        Calculate the matrix once retrieve from the cache on subsequent calls.
        """
        return tuple((-x, y, c) for x, y, c in self.value)


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


def lau_arce_gallagher(image, bit_depth, hysteresis_constant: float = 1, serpentine: bool = True):
    return green_noise_halftone(
        image, bit_depth, (DiffusionMatrix.FLOYD_STEINBERG.hysteresis, DiffusionMatrix.FLOYD_STEINBERG.hysteresis_reversed,
                           DiffusionMatrix.STUCKI.value, DiffusionMatrix.STUCKI.reversed), hysteresis_constant, serpentine
    )


@cfunc('boolean(int64, int64, int64, int64)')
def check_bounds(x, y, width, height):
    return 0 <= x < width and 0 <= y < height


@cfunc('float64(float64)')
def clamp(color):
    return max(0, min(color, 255))


@cfunc('float64(float64, int64)')
def quantize_uniform(color: float, palette_size: float) -> int:
    return round(color / 255.0 * palette_size) / palette_size * 255.0


@njit
def dither(image: np.ndarray, bit_depth: int, matrix: Tuple) -> np.ndarray:
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
                    if check_bounds(x + u, y + v, width, height):
                        image[y + v, x + u, c] += quant_error * k
    return image


@njit
def green_noise_halftone(image: np.ndarray, bit_depth: int, matrix: Tuple, H: int, serpentine: bool = True) -> np.ndarray:
    height, width, channels = image.shape
    palette_size = (1 << bit_depth) - 1

    if serpentine:
        hysteresis, hysteresis_reversed, diffusion, diffusion_reversed = matrix
        hysteresis_matrices = hysteresis, hysteresis_reversed
        diffusion_matrices = diffusion, diffusion_reversed
    else:
        hysteresis, _, diffusion, _ = matrix
        hysteresis_matrices = hysteresis, hysteresis
        diffusion_matrices = diffusion, diffusion

    for y in range(height):
        aim = y % 2
        range_args = ((0, width, 1), (width, 0, -1))
        for x in range(*range_args[aim]):
            for c in range(channels):
                input_color = image[y, x, c]
                for u, v, k in hysteresis_matrices[aim]:
                    if check_bounds(x + u, y + v, width, height):
                        image[y, x, c] += H * k * image[y + v, x + u, c]
                color = image[y, x, c]
                quant_color = quantize_uniform(color, palette_size)
                image[y, x, c] = quant_color
                quant_error = quant_color - input_color
                for u, v, k in diffusion_matrices[aim]:
                    if check_bounds(x + u, y + v, width, height):
                        image[y + v, x + u, c] -= quant_error * k
                image[y, x, c] = clamp(image[y, x, c])

    return image
