from itertools import product


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
        return x @ q @ x + offset

    @staticmethod
    def get_matrix_dict_repr(q):
        """
        Returns dict representation of a QUBO matrix.
        :param q:
            QUBO matrix
        :return:
            the input matrix as a dict object
        """
        dict_repr = {}
        for i, j in product(range(len(q)), repeat=2):
            if q[i, j] != 0:
                dict_repr[(i, j)] = q[i, j]
        return dict_repr

    @staticmethod
    def get_solution_vector_repr(d):
        """
        Returns vector representation of solution vector.
        :param q:
            solution vector
        :return:
            the input solution as a vector
        """
        return [d[i] for i in range(len(d))]

    @staticmethod
    def solve(q, offset=0, target=None, timeout=60):
        """
        Returns solutions to a given QUBO problem.
        :param q:
            QUBO matrix
        :param offset:
            offset
         : param target:
             target energy as stopping criterion
        :return: solutions, energies
        """
        from dwave_qbsolv import QBSolv
        import warnings

        dq = Utils.get_matrix_dict_repr(q)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="QBSolv is deprecated")
            response = QBSolv().sample_qubo(dq, target=target, timeout=timeout)

        samples = [Utils.get_solution_vector_repr(d) for d in response.samples()]
        energies = [e + offset for e in response.data_vectors["energy"]]
        return samples, energies
