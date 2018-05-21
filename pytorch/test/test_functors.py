import unittest
import numpy as np

from functors import flip_bits_fn


class TestLayers(unittest.TestCase):

    def test_that_bit_flip_with_probability_of_zero_changes_no_bits(self):
        sample_array = np.random.rand(3, 3).astype('float32')

        differences = sample_array[np.not_equal(sample_array, flip_bits_fn(sample_array, 0))]

        self.assertTrue(differences.size == 0)

    def test_that_bit_flip_with_probability_of_one_changes_all_bits(self):
        sample_array = np.random.rand(3, 3).astype('float32')

        differences = sample_array[np.not_equal(sample_array, flip_bits_fn(sample_array, 1))]

        self.assertTrue(differences.size == sample_array.size)

