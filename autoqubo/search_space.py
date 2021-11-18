class SearchSpace:
    """
    Provides methods for describing a search space that is not binary. Provides methods for transforming elements of
    the original search space into binary vectors and the other way around.
    """
    def __init__(self, desc=None):
        self.desc = []
        self.size = 0
        if desc is not None:
            self.add_all(desc)

    def add(self, label, var_type, var_size):
        """
        Add a new decision variable to the search space.
        :param label:
        :param var_type:
        :param var_size:
        :return:
        """
        self.desc.append((label, var_type, var_size))
        self.size += var_size

    def add_all(self, desc):
        """
        Add multiple new decision variables to the search space from a list.
        :param desc:
        :return:
        """
        for args in desc:
            self.add(*args)

    def decode(self, x):
        """
        Decode a bitstring into a member of the search space represented as a list.
        :param x:
        :return:
        """
        values = []
        k = 0
        for label, decoding, size in self.desc:
            values.append(decoding.decode(x[k:k+size]))
            k += size
        return values

    def decode_dict(self, x):
        """
        Decode a bitstring into a member of the search space represented as a dict.
        :param x:
        :return:
        """
        values = {}
        k = 0
        for label, decoding, size in self.desc:
            values[label] = decoding.decode(x[k:k+size])
            k += size
        return values

    def encode(self, values):
        """
        Encode a member of the search space into a bitstring.
        :param values:
        :return:
        """
        bitstring = []
        for label, decoding, size in self.desc:
            bitstring += decoding.encode(values[label], size)
        return bitstring

    def call_binary(self, f, x):
        """
        Call a function using a binary input.
        :param f:
        :param x:
        :return:
        """
        return f(*self.decode(x))

    def wrap_binary(self, f):
        """
        Get a function that accepts binary input.
        :param f:
        :return:
        """
        def binary(x):
            return self.call_binary(f, x)
        return binary
