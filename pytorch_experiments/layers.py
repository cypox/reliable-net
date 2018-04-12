import torch

from pytorch_experiments.functors import flip_bits_fn


class BitflipLayer(torch.nn.Module):
    def __init__(self, probability):
        super(BitflipLayer, self).__init__()
        self._probability = probability

    def forward(self, input):
        flipped = flip_bits_fn(input.data.numpy(), self._probability)

        return torch.from_numpy(flipped)

    def backward(self, grad_output):
        return grad_output