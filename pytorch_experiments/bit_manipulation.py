import numpy as np
import bitstring


def flip_bits(x: float, probability: float):
    '''
    Flip bits from the bitwise representation of a float with a given
    probability
    :param x: float
    :param probability: probability of flipping abit
    :return: bit flipped float
    '''
    bit_array = bitstring.BitArray(float=x, length=32)
    original_bits = np.array(bit_array).astype(bool)

    bits_to_be_flipped = np.random.uniform(0, 1, len(original_bits)) < probability
    original_bits[bits_to_be_flipped] = np.invert(original_bits[bits_to_be_flipped])

    bit_array.bin = ''.join(['0' if not bit else '1' for bit in original_bits])

    return bit_array.float


def flip_one_bit(x: float):
    """
    Flip one bit from a float
    :param x: float value for a weight
    :return:
    """
    bit_array = bitstring.BitArray(float=x, length=32)
    original_bits = np.array(bit_array).astype(bool)

    bit_to_flip = int(np.random.uniform(0, 1) * 32)
    original_bits[bit_to_flip] = np.invert(original_bits[bit_to_flip])

    bit_array.bin = ''.join(['0' if not bit else '1' for bit in original_bits])

    return bit_array.float