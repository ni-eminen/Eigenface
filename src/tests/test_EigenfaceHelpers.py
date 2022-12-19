import unittest
from src.helpers import EigenfaceHelpers, negative_vector, sum_of_vectors, scalar_multiply_vector, euclidean_distance, \
    manhattan_distance, hamming_distance, KNN_prediction, n_of_the_same
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

    def test_euclidean_distance(self):
        vector1 = [[1, 1], [0, 0], [10, 15]]
        self.assertListEqual(euclidean_distance(np.array(vector1), np.array([0, 0])).tolist(),
                             [1.4142135623730951, 0, 18.027756377319946])
        self.assertEqual(euclidean_distance(np.array(vector1), np.array([0, 0, 0])), None)

    def test_manhattan_distance(self):
        matrix = [[1, 1], [0, 0], [0, 1]]
        vector1 = np.array([1, 2])

        self.assertListEqual(manhattan_distance(matrix, vector1).tolist(), [1.0, 3.0, 2.0])
        self.assertEqual(manhattan_distance(np.array(matrix), np.array([0, 0, 0])), None)

    def test_hamming_distance(self):
        matrix = [[1, 1], [0, 0], [0, 1]]
        vector1 = np.array([1, 2])

        self.assertListEqual(hamming_distance(matrix, vector1).tolist(), [0.5, 1.0, 1.0])
        self.assertEqual(hamming_distance(np.array(matrix), np.array([0, 0, 0])), None)

    def test_KNN_prediction(self):
        # Here 2 is the lowest and the most common id in the targets.
        norms = [10, 20, 9, 40]
        targets = [1, 2, 1, 4]
        sample_size = 3
        threshold = 2
        idx_train = [0, 1, 2, 3]
        predicted = KNN_prediction(norms, sample_size, threshold, idx_train, targets)
        self.assertEqual(predicted, 2)

    def test_n_of_the_same(self):
        arr = [1, 2, 3, 4, 5, 6, 1, 3, 5, 2, 5, 0]
        arr2 = [1, 2, 3]
        n = 3
        self.assertEqual(n_of_the_same(arr, n), 5)
        self.assertEqual(n_of_the_same(arr2, n), None)
        self.assertEqual(n_of_the_same([], n), None)
        self.assertEqual(n_of_the_same([1], n), None)
