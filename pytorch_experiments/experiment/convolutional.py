import numpy as np

import pickle

from tqdm import tqdm
import torch

from torch.autograd import Variable

from pytorch_experiments.layers import WeightBitflipByCountLayer


class NoisyConvolution(object):
    """
    Performs noisy convolution flipping bits in the output
    """
    def __init__(self, channels_in, channels_out, input_shape):
        self._convolution = torch.nn.Conv2d(channels_in,
                                            channels_out,
                                            input_shape,
                                            bias=False)

        torch.nn.init.xavier_uniform(self._convolution.weight)

    def noisy_activation(self,
                         input: torch.autograd.Variable,
                         weight_count: int):
        """
        Build a model built by a convolutional layer and
        a bitflip layer and return its output
        :param input: Input tensor
        :param error_probability: Bitflip probability
        :return: Output tensor
        """
        return torch.nn.Sequential(self._convolution,
                                   WeightBitflipByCountLayer(weight_count))(input)


def sample_image(input_shape):
    """
    Create an array of the given shape and populate it with random samples from
    a uniform distribution over (1, 1)
    :return:
    """
    buffer =  255 * np.random.rand(1,
                                   input_shape[0],
                                   input_shape[1],
                                   input_shape[2]).astype('float32')

    return Variable(torch.from_numpy(buffer)).float()

def sample_input(input_shape):
    """
    Create an array of the given shape and populate it with random samples from
    a uniform distribution over (1, 1)
    :return:
    """
    buffer =  2 * np.random.rand(1,
                                 1,
                                 input_shape[0],
                                 input_shape[1]).astype('float32') - 1

    return Variable(torch.from_numpy(buffer)).float()


def cnn_experiment(num_channels, image_size, kernel_size, weights_to_flip):
    """
    Run the CNN experiment and report delta between golden and dirty run
    :param num_channels: Number of channels for the convolutional layer
    :param image_size: Tuple representing the size of an image (WxH)
    :param kernel_size: Tuple representing the size of the kernel (WxH)
    :param weights_to_flip: Error rate for bitflip
    :return:
    """
    sample_array = sample_image((num_channels, image_size[0], image_size[1]))

    noisy_convolution = NoisyConvolution(1, num_channels, kernel_size)

    golden_run = noisy_convolution.noisy_activation(sample_array, 0)
    dirty_run = noisy_convolution.noisy_activation(sample_array, weights_to_flip)

    return golden_run - dirty_run

def main():

    image_results = []
    for _ in tqdm(range(1000000)):
        image_results.append(cnn_experiment(1, (28, 28), (3, 3), 1))

    pickle.dump(image_results, open('image_results.pkl', 'wb'))


if __name__ == '__main__':
    main()