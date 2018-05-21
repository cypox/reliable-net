import torch

from pytorch_experiments.functors import flip_bits_by_probability, flip_weights_by_count


class BitflipByProbabilityLayer(torch.nn.Module):
    """
    Represents a layer for flipping bits at a given probability
    """
    def __init__(self, probability: float):
        super(BitflipByProbabilityLayer, self).__init__()
        self._probability = probability

    def forward(self, input):
        """
        Forward pass
        :param input: input tensor
        :return: output tensor
        """
        flipped = flip_bits_by_probability(input.data.numpy(), self._probability)

        return torch.from_numpy(flipped)

    def backward(self, grad_output):
        """
        Backward pass
        :param grad_output: output from previous layer
        :return: forward gradient untouched
        """
        return grad_output


class WeightBitflipByCountLayer(torch.nn.Module):
    """
    Represents a layer for flipping one bit to a given number of weights
    """
    def __init__(self, count: int):
        super(WeightBitflipByCountLayer, self).__init__()
        self._count = count

    def forward(self, input):
        """
        Forward pass
        :param input: input tensor
        :return: output tensor
        """
        flipped = flip_weights_by_count(input.data.numpy(), self._count)

        return torch.from_numpy(flipped)

    def backward(self, grad_output):
        """
        Backward pass
        :param grad_output: output from previous layer
        :return: forward gradient untouched
        """
        return grad_output