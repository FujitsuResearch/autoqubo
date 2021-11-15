from collections import namedtuple


Type = namedtuple('Type', 'decode encode')


def uint_decode(bitstring):
    return sum(v * 2 ** i for (i, v) in enumerate(bitstring))


def uint_encode(value, binary_size):
    return [(1 if value & 2 ** i else 0) for i in range(binary_size)]


class Binarization:
    uint = Type(uint_decode, uint_encode)
