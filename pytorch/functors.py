import numpy as np

from bit_manipulation import flip_bits


def flip_bits_fn(x: np.ndarray, probability: float):
    """
    Vectorize the flip bits call
    :param x: float
    :param probability: probability to flip bits
    :return: vectorized mapped function call result
    """
    vfunc = np.vectorize(flip_bits)

    return vfunc(x.flatten(), probability).reshape(x.shape)