import torch

from pytorch.functors import flip_bits_fn


class BitflipLayer(torch.nn.Module):
    """
    Represents a layer for flipping bits at a given probability
    """
    def __init__(self, probability):
        super(BitflipLayer, self).__init__()
        self._probability = probability

    def forward(self, input):
        """
        Forward pass
        :param input: input tensor
        :return: output tensor
        """
        flipped = flip_bits_fn(input.data.numpy(), self._probability)

        return torch.from_numpy(flipped)

    def backward(self, grad_output):
        """
        Backward pass
        :param grad_output: output from previous layer
        :return: forward gradient untouched
        """
        return grad_output