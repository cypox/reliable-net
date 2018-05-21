import unittest
import numpy as np

from pytorch_experiments.functors import flip_weights_by_count, flip_bits_by_probability


class TestLayers(unittest.TestCase):

    def test_that_bit_flip_with_probability_of_zero_changes_no_bits(self):
        sample_array = np.random.rand(3, 3).astype('float32')

        differences = sample_array[np.not_equal(sample_array, flip_bits_by_probability(sample_array, 0))]

        self.assertTrue(differences.size == 0)

    def test_that_bit_flip_with_probability_of_one_changes_all_bits(self):
        sample_array = np.random.rand(3, 3).astype('float32')

        differences = sample_array[np.not_equal(sample_array, flip_bits_by_probability(sample_array, 1))]

        self.assertTrue(differences.size == sample_array.size)

    def test_that_weight_bit_flip_with_count_of_one_changes_one_weight(self):
        sample_array = np.random.rand(3, 3).astype('float32')

        dirty_array = flip_weights_by_count(sample_array, 1)

        self.assertTrue(dirty_array.size == sample_array.size)
        comparison = sum([x == y for x, y in zip(sample_array.flatten(), dirty_array.flatten())])

        self.assertEqual(comparison, sample_array.size - 1)


    def test_that_weight_bit_flip_with_count_of_three_changes_three_weights(self):
        sample_array = np.random.rand(3, 3).astype('float32')

        dirty_array = flip_weights_by_count(sample_array, 3)

        self.assertTrue(dirty_array.size == sample_array.size)
        comparison = sum([x == y for x, y in zip(sample_array.flatten(), dirty_array.flatten())])

        self.assertEqual(comparison, sample_array.size - 3)
