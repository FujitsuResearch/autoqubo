from collections import namedtuple
import numpy as np


Type = namedtuple('Type', 'decode encode')


def uint_decode(bitstring):
    return sum(v * 2 ** i for (i, v) in enumerate(bitstring))


def uint_encode(value, binary_size):
    return [(1 if value & 2 ** i else 0) for i in range(binary_size)]


class Binarization:
    """
    Provides a namespace containing objects describing various types of decision variables that can be transformed into
    binary.
    Contains: `uint`.
    """
    uint = Type(uint_decode, uint_encode)

    @staticmethod
    def get_uint_vector_type(uint_size, n):

        def encode(value, binary_size):
            bitstring = []
            for i in range(n):
                bitstring += uint_encode(value[i], uint_size)
            return bitstring

        def decode(bitstring):
            elems = []
            p = 0
            for i in range(n):
                elems.append(uint_decode(bitstring[p:p+uint_size]))
                p += uint_size
            return np.array(elems)
        t = Type(decode, encode)
        return t
