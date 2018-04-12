import numpy as np

from pytorch_experiments.bit_manipulation import flip_bits


def flip_bits_fn(x: np.ndarray, probability: float):
    vfunc = np.vectorize(flip_bits)

    return vfunc(x.flatten(), probability).reshape(x.shape)