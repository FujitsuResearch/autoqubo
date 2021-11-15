class SearchSpace:
    def __init__(self, desc=None):
        self.desc = []
        self.size = 0
        if desc is not None:
            self.add_all(desc)

    def add(self, label, var_type, var_size):
        self.desc.append((label, var_type, var_size))
        self.size += var_size

    def add_all(self, desc):
        for args in desc:
            self.add(*args)

    def decode(self, x):
        values = []
        k = 0
        for label, decoding, size in self.desc:
            values.append(decoding.decode(x[k:k+size]))
            k += size
        return values

    def decode_dict(self, x):
        values = {}
        k = 0
        for label, decoding, size in self.desc:
            values[label] = decoding.decode(x[k:k+size])
            k += size
        return values

    def encode(self, values):
        bitstring = []
        for label, decoding, size in self.desc:
            bitstring += decoding.encode(values[label], size)
        return bitstring

    def call_binary(self, f, x):
        return f(*self.decode(x))

    def wrap_binary(self, f):
        def binary(x):
            return self.call_binary(f, x)
        return binary
