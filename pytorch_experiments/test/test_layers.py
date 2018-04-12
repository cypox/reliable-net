import unittest

import numpy as np
import torch
from torch.autograd import Variable

from pytorch_experiments.layers import BitflipLayer


class TestFunctors(unittest.TestCase):

    def test_that_convolutoinal_layer_shows_reasonable_output(self):

        sample_array = Variable(torch.from_numpy(np.ones((1,1,5,5)))).float()

        convolutional_layer = torch.nn.Conv2d(1, 1, (3, 3), bias=False)
        convolutional_layer.weight.data.fill_(1)

        prediction = convolutional_layer(sample_array).data.numpy()

        self.assertTrue(prediction.shape == (1, 1, 3, 3))
        self.assertTrue(prediction.max() == 9)
        self.assertTrue(prediction.min() == 9)

    def test_that_BitFlipLayer_with_probability_of_zero_changes_no_bits(self):
        sample_array = Variable(torch.from_numpy(np.ones((1, 1, 5, 5)))).float()

        convolutional_layer = torch.nn.Conv2d(1, 1, (3, 3), bias=False)
        convolutional_layer.weight.data.fill_(1)

        prediction = convolutional_layer.forward(sample_array)

        bit_flip_layer = BitflipLayer(0)

        prediction = bit_flip_layer(prediction)

        self.assertTrue(prediction.shape == (1, 1, 3, 3))
        self.assertTrue(prediction.max() == 9)
        self.assertTrue(prediction.min() == 9)

    def test_that_BitFlipLayer_with_probability_of_one_changes_all_bits(self):
        sample_array = Variable(torch.from_numpy(np.ones((1, 1, 1, 1)))).float()

        convolutional_layer = torch.nn.Conv2d(1, 1, (1, 1), bias=False)
        convolutional_layer.weight.data.fill_(1)

        prediction = convolutional_layer.forward(sample_array)

        bit_flip_layer = BitflipLayer(1)

        prediction = bit_flip_layer(prediction).numpy()

        self.assertTrue(prediction.shape == (1, 1, 1, 1))
        self.assertTrue(prediction.max() + 4  < 10e-5)
        self.assertTrue(prediction.min() + 4  < 10e-5)
