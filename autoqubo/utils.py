class Utils:
    @staticmethod
    def energy(q, x, offset=0):
        """
        Calculate the QUBO energy for a given binary solution.
        :param q:
        :param x:
        :param offset:
        :return:
        """
        return x@q@x + offset
