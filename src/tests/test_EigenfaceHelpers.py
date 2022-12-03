import unittest
from src.helpers import EigenfaceHelpers, negative_vector, sum_of_vectors, scalar_multiply_vector
import numpy as np
import PIL
import os


class TestEigenHelpers(unittest.TestCase):
    def setUp(self) -> None:
        self.eighelpers = EigenfaceHelpers((64, 64))
        self.img = PIL.Image.fromarray(np.zeros((120, 120)))

    def test_crop_image(self):
        newimg = self.eighelpers.crop_image(self.img)
        self.assertEqual((64, 64), np.array(newimg).shape)

    def test_img_to_vector(self):
        img = os.path.dirname(os.path.abspath('.'))
        print(img)
        vector = self.eighelpers.img_to_vector(img + '/eigenface/src/tests/test_images/George_W_Bush_0005.jpg')
        self.assertEqual(vector.shape, (4096,))

    def test_vector_to_img(self):
        newimg = self.eighelpers.vector_to_img(np.zeros(4096))
        self.assertEqual((64, 64), newimg.size)

    def test_sum_of_vectors(self):
        vector1 = [1, 2, 3]
        vector2 = [1, 2, 3]
        sum = sum_of_vectors([vector2, vector1])
        self.assertEqual(sum[0], 2)

    def test_scalar_m_vector(self):
        vector1 = [1, 2, 3]
        self.assertEqual(scalar_multiply_vector(2, vector1)[0], 2)

    def test_neg_vector(self):
        vector1 = [1, 2, 3]
        self.assertEqual(negative_vector(vector1)[0], -1)
