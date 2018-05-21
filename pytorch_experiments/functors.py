import numpy as np

from pytorch_experiments.bit_manipulation import flip_bits, flip_one_bit


def flip_bits_by_probability(x: np.ndarray, probability: float):
    """
    Vectorize the flip bits call
    :param x: float
    :param probability: probability to flip bits
    :return: vectorized mapped function call result
    """
    vfunc = np.vectorize(flip_bits)

    return vfunc(x.flatten(), probability).reshape(x.shape)


def flip_weights_by_count(x: np.ndarray, count: int):
    """
    Randomly select some weights that will have one bit flipped
    :param x: float
    :param probability: probability to flip bits
    :return: vectorized mapped function call result
    """
    flattened = x.flatten()
    weights_to_flip = np.random.choice(flattened.size, count, replace=False)

    for i in weights_to_flip:
        flattened[i] = flip_one_bit(flattened[i])

    return flattened.reshape(x.shape)