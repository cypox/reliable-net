import numpy as np
import bitstring


def flip_bits(x: float, probability: float):
    bit_array = bitstring.BitArray(float=x, length=32)
    original_bits = np.array(bit_array).astype(bool)

    bits_to_be_flipped = np.random.uniform(0, 1, len(original_bits)) < probability
    original_bits[bits_to_be_flipped] = np.invert(original_bits[bits_to_be_flipped])

    bit_array.bin = ''.join(['0' if not bit else '1' for bit in original_bits])

    return bit_array.float