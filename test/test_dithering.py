import unittest

import numpy as np
from PIL import Image

from dithering import (
    DiffusionMatrix,
    floyd_steinberg,
    jarvis_judice_ninke,
    sierra,
    lau_arce_gallagher
)


class TestDithering(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def load_image(self, input_path):
        image = Image.open(input_path)
        print(f'Loaded image with {input_path = }')
        return image

    @classmethod
    def save_image(self, image, output_path):
        image.save(output_path)
        print(f'Saved image with {output_path = }')

    def get_dithered_image(self, image, bit_depth, func, **kwargs):
        input_data = np.array(image).astype(np.float32)
        output_data = func(input_data, bit_depth, **kwargs).astype(np.uint8)
        return output_data

    def verify_palette_size(self, image, bit_depth, num_channels=3) -> bool:
        palette_size = 1
        for i in range(num_channels):
            print(np.unique(image[:, :, i]))
            print(np.unique(image[:, :, i]).size)
            palette_size *= np.unique(image[:, :, i]).size
        expected_palette_size = (1 << bit_depth) ** num_channels
        print(expected_palette_size, palette_size)
        return palette_size <= expected_palette_size

    def test_real_palette_size(self):
        input_path = './images/hatsune_miku.png'
        image = self.load_image(input_path)
        bit_depth = 2
        output_data = self.get_dithered_image(
            image, bit_depth, floyd_steinberg)
        self.assertTrue(self.verify_palette_size(output_data, bit_depth))

        output_data = self.get_dithered_image(
            image, bit_depth, lau_arce_gallagher,  hysteresis_constant=1/2.5)
        self.assertTrue(self.verify_palette_size(output_data, bit_depth))

    def test_visual(self):
        input_path = './images/hatsune_miku.png'
        output_path = './images/hatsune_miku_dithered_2_bits.png'
        image = self.load_image(input_path)
        dithered_image = self.get_dithered_image(image, 2, floyd_steinberg)
        self.save_image(Image.fromarray(dithered_image), output_path)

        input_path = './images/kuudere.png'
        output_path = './images/kuudere_dithered_2_bits.png'
        image = self.load_image(input_path)
        dithered_image = self.get_dithered_image(image, 2, floyd_steinberg)
        self.save_image(Image.fromarray(dithered_image), output_path)

        input_path = './images/smile.png'
        output_path = './images/smile_dithered_1_bit.png'
        image = self.load_image(input_path)
        dithered_image = self.get_dithered_image(image, 1, floyd_steinberg)
        self.save_image(Image.fromarray(dithered_image), output_path)

        input_path = './images/キャンディーポット.png'
        output_path = './images/キャンディーポット_dithered_1_bit.png'
        image = self.load_image(input_path)
        dithered_image = self.get_dithered_image(image, 1, sierra)
        self.save_image(Image.fromarray(dithered_image), output_path)

    def test_hysteresis_generation(self):
        FLOYD_STEINBERG_HYSTERESIS = (
            (-1, 0, 0.4375),
            (1, -1, 0.1875),
            (0, -1, 0.3125),
            (-1, -1, 0.0625)
        )
        self.assertEqual(
            DiffusionMatrix.FLOYD_STEINBERG.hysteresis, FLOYD_STEINBERG_HYSTERESIS)

    def test_green_noise_halftone(self):
        input_path = './images/smile.png'
        output_path = './images/smile_green_noise_dithered_1_bit.png'
        image = self.load_image(input_path)
        dithered_image = self.get_dithered_image(
            image, 1, lau_arce_gallagher, hysteresis_constant=1/2.5)
        self.save_image(Image.fromarray(dithered_image), output_path)

        input_path = './images/hatsune_miku.png'
        output_path = './images/hatsune_miku_green_noise_dithered_2_bits.png'
        image = self.load_image(input_path)
        dithered_image = self.get_dithered_image(
            image, 2, lau_arce_gallagher, hysteresis_constant=1/2.5)
        self.save_image(Image.fromarray(dithered_image), output_path)


if __name__ == '__main__':
    unittest.main()
