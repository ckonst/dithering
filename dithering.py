# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 15:12:44 2022

@author: Christian Konstantinov
"""

from functools import lru_cache
from enum import Enum
from typing import Tuple

import numpy as np
from numba import njit, cfunc
from PIL import Image

#TODO: refactor dithering functions to a single one, utlize more of numba's cfuncs. 

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
    def reverse_hysteresis(self):
        """
        Return the matrix rotated 90 degrees for hysteresis filter preturbation.
        Calculate the matrix once retrieve from the cache on subsequent calls.
        """
        return tuple((x, -y, c) for x, y, c in self.value)

    @property
    @lru_cache(maxsize=1)
    def reverse(self):
        """
        Return the matrix flipped horizontally for odd-row serpentine diffusion.
        Calculate the matrix once retrieve from the cache on subsequent calls.
        """
        return tuple((-x, y, c) for x, y, c in self.value)

    @property
    @lru_cache(maxsize=1)
    def bidirectional(self):
        return (self.value, self.reverse)

    @property
    @lru_cache(maxsize=1)
    def bidirectional_hysteresis(self):
        return (self.hysteresis, self.reverse_hysteresis)


def atkinson(image: np.ndarray, bit_depth, serpentine: bool = False) -> np.ndarray:
    return dither(image, DiffusionMatrix.ATKINSON, bit_depth, serpentine)


def burkes(image: np.ndarray, bit_depth: int = 1, serpentine: bool = False) -> np.ndarray:
    return dither(image, DiffusionMatrix.BURKES, bit_depth, serpentine)


def floyd_steinberg(image: np.ndarray, bit_depth: int = 1, serpentine: bool = False) -> np.ndarray:
    return dither(image, DiffusionMatrix.FLOYD_STEINBERG, bit_depth, serpentine)


def false_floyd_steinberg(image: np.ndarray, bit_depth: int = 1, serpentine: bool = False) -> np.ndarray:
    return dither(image, DiffusionMatrix.FALSE_FLOYD_STEINBERG, bit_depth, serpentine)


def jarvis_judice_ninke(image: np.ndarray, bit_depth: int = 1, serpentine: bool = False) -> np.ndarray:
    return dither(image, DiffusionMatrix.JARVIS_JUDICE_NINKE, bit_depth, serpentine)


def stucki(image: np.ndarray, bit_depth: int = 1, serpentine: bool = False) -> np.ndarray:
    return dither(image, DiffusionMatrix.STUCKI, bit_depth, serpentine)


def sierra(image: np.ndarray, bit_depth: int = 1, serpentine: bool = False) -> np.ndarray:
    return dither(image, DiffusionMatrix.SIERRA, bit_depth, serpentine)


def two_row_sierra(image: np.ndarray, bit_depth: int = 1, serpentine: bool = False) -> np.ndarray:
    return dither(image, DiffusionMatrix.TWO_ROW_SIERRA, bit_depth, serpentine)


def sierra_lite(image: np.ndarray, bit_depth: int = 1, serpentine: bool = False) -> np.ndarray:
    return dither(image, DiffusionMatrix.SIERRA_LITE, bit_depth, serpentine)


def lau_arce_gallagher(
    image: np.ndarray,
    hysteresis_constant: float = 1,
    threshold: float = 0.0,
    bit_depth: int = 1,
    serpentine: bool = False
) -> np.ndarray:
    return green_noise_halftone(
        image, bit_depth=bit_depth, H=hysteresis_constant,
        threshold=threshold, serpentine=serpentine
    )


def dither(
    image: np.ndarray,
    matrix: DiffusionMatrix = DiffusionMatrix.FLOYD_STEINBERG,
    bit_depth: int = 1,
    serpentine: bool = False
) -> np.ndarray:
    return _serpentine_dither(image, matrix.bidirectional, bit_depth) if serpentine else _raster_scan_dither(image, matrix.value, bit_depth)


def green_noise_halftone(
    image: np.ndarray,
    diffusion_matrix: DiffusionMatrix = DiffusionMatrix.STUCKI,
    hysteresis_matrix: DiffusionMatrix = DiffusionMatrix.FLOYD_STEINBERG,
    H: int = 1.0,
    threshold: float = 0.0,
    bit_depth: int = 1,
    serpentine: bool = False
) -> np.ndarray:
    return _serpentine_green_noise_halftone(
        image, diffusion_matrix.bidirectional, hysteresis_matrix.bidirectional_hysteresis,
        H, threshold, bit_depth
    ) if serpentine else _raster_scan_green_noise_halftone(
        image, diffusion_matrix.value, hysteresis_matrix.hysteresis,
        H, threshold, bit_depth
    )


@njit
def _raster_scan_dither(image: np.ndarray, matrix: Tuple, bit_depth: int) -> np.ndarray:
    height, width, channels = image.shape
    palette_size = (1 << bit_depth) - 1

    for y in range(height):
        for x in range(width):
            for c in range(channels):
                color = image[y, x, c]
                quant_color = quantize_uniform(color, palette_size)
                image[y, x, c] = clamp(quant_color)
                quant_error = color - quant_color
                for u, v, k in matrix:
                    if check_bounds(x + u, y + v, width, height):
                        image[y + v, x + u, c] += quant_error * k
    return image


@njit
def _serpentine_dither(image: np.ndarray, matrix: Tuple, bit_depth: int) -> np.ndarray:
    height, width, channels = image.shape
    palette_size = (1 << bit_depth) - 1

    diffusion, diffusion_reverse = matrix
    diffusion_matrices = (diffusion, diffusion_reverse)
    direction = ((0, width, 1), (width, 0, -1))

    for y in range(height):
        aim = y % 2
        for x in range(*direction[aim]):
            for c in range(channels):
                color = image[y, x, c]
                quant_color = quantize_uniform(color, palette_size)
                image[y, x, c] = clamp(quant_color)
                quant_error = color - quant_color
                for u, v, k in diffusion_matrices[aim]:
                    if check_bounds(x + u, y + v, width, height):
                        image[y + v, x + u, c] += quant_error * k
    return image


@njit
def _raster_scan_green_noise_halftone(image: np.ndarray, diffusion_matrix: Tuple, hysteresis_matrix: Tuple, H: int, threshold: float, bit_depth: int) -> np.ndarray:
    height, width, channels = image.shape
    palette_size = (1 << bit_depth) - 1

    for y in range(height):
        for x in range(width):
            for c in range(channels):
                input_color = image[y, x, c]
                hysteresis_color = input_color
                for u, v, k in hysteresis_matrix:
                    if check_bounds(x + u, y + v, width, height) and image[y + v, x + u, c] >= 255.0 * threshold:
                        hysteresis_color += H * k * image[y + v, x + u, c]
                quant_color = quantize_uniform_linear(
                    hysteresis_color, palette_size)
                image[y, x, c] = clamp(quant_color)
                quant_error = input_color - quant_color
                for u, v, k in diffusion_matrix:
                    if check_bounds(x + u, y + v, width, height):
                        image[y + v, x + u, c] += quant_error * k
    return image


@njit
def _serpentine_green_noise_halftone(image: np.ndarray, diffusion_matrices: Tuple, hysteresis_matrices: Tuple, H: int, threshold: float, bit_depth: int) -> np.ndarray:
    height, width, channels = image.shape
    palette_size = (1 << bit_depth) - 1

    direction = ((0, width, 1), (width, 0, -1))

    for y in range(height):
        aim = y % 2
        for x in range(*direction[aim]):
            for c in range(channels):
                input_color = image[y, x, c]
                hysteresis_color = input_color
                for u, v, k in hysteresis_matrices[aim]:
                    if check_bounds(x + u, y + v, width, height) and image[y + v, x + u, c] >= 255.0 * threshold:
                        hysteresis_color += H * k * image[y + v, x + u, c]
                quant_color = quantize_uniform_linear(
                    hysteresis_color, palette_size)
                image[y, x, c] = clamp(quant_color)
                quant_error = input_color - quant_color
                for u, v, k in diffusion_matrices[aim]:
                    if check_bounds(x + u, y + v, width, height):
                        image[y + v, x + u, c] += quant_error * k
    return image


@cfunc('boolean(int64, int64, int64, int64)')
def check_bounds(x: int, y: int, width: int, height: int) -> bool:
    return 0 <= x < width and 0 <= y < height


@cfunc('float64(float64)')
def clamp(color: float) -> float:
    return max(0, min(color, 255))


@cfunc('float64(float64, float64)')
def gamma_correct(color: float, gamma: float) -> float:
    return color / 12.92 if color < 0.04045 else ((color + 0.055) / 1.055) ** gamma


@cfunc('float64(float64, int64)')
def quantize_uniform_linear(color: float, palette_size: float) -> int:
    return round(gamma_correct(color / 255.0, 2.4) * palette_size) / palette_size * 255.0


@cfunc('float64(float64, int64)')
def quantize_uniform(color: float, palette_size: float) -> int:
    return round(color / 255.0 * palette_size) / palette_size * 255.0
