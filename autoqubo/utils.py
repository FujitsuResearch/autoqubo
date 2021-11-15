class Utils:
    @staticmethod
    def energy(q, x, offset=0):
        return x@q@x + offset
