import numpy as np
from typing import Callable, Optional, Tuple


class SamplingCompiler:
    """
    Provides .generate_qubo_matrix() method that allows to transform a function into a QUBO model.
    """

    @staticmethod
    def _indices_iterator(input_size):
        yield tuple()

        for i in range(input_size):
            yield (i,)

        for i in range(input_size):
            for j in range(i+1, input_size):
                yield (i, j)

    @staticmethod
    def _new_training_sample(input_size, idx):
        sample = [0] * input_size
        for i in idx:
            sample[i] = 1
        return sample

    @staticmethod
    def _get_training_samples(input_size):
        return (SamplingCompiler._new_training_sample(input_size, idx)
                for idx in SamplingCompiler._indices_iterator(input_size))

    @staticmethod
    def _generate_training_output(fitness_function, input_size):
        return (fitness_function(sample) for sample in SamplingCompiler._get_training_samples(input_size))

    @classmethod
    def _generate_qubo_coefficients(cls, fitness_function, input_size):
        coefficients = []
        for output, index in zip(cls._generate_training_output(fitness_function, input_size),
                                 cls._indices_iterator(input_size)):
            coefficients.append(output - (0 if len(index) < 2 else sum(coefficients[i+1] for i in index)) -
                                (0 if index == tuple() else coefficients[0]))
        return coefficients

    @staticmethod
    def _qubo_matrix(coefficients, input_size, dtype=np.float64):
        qubo = np.zeros((input_size, input_size), dtype=dtype)
        k = 0
        for idx in SamplingCompiler._indices_iterator(input_size):
            if len(idx) == 1:
                qubo[idx[0], idx[0]] = coefficients[k]
            if len(idx) == 2:
                qubo[idx[0], idx[1]] = coefficients[k]
            k += 1
        return qubo, coefficients[0]

    @classmethod
    def generate_qubo_matrix(cls,
                             fitness_function: Callable,
                             input_size: int,
                             searchspace: Optional['SearchSpace'] = None) -> Tuple[np.array, int]:
        """
        Generates a QUBO matrix for a given function.
        :param fitness_function: Callable
            Function to be compiled.
        :param input_size: int
            number of binary variables in the function input.
        :param searchspace: SearchSpace
            Optional parameter describing the arguments of the funfion.
        :return: Q, c
            Q: QQUBO matrix
            c: offset / constant term
        """
        if searchspace is None:
            return cls._qubo_matrix(cls._generate_qubo_coefficients(fitness_function, input_size), input_size)
        else:
            return cls._qubo_matrix(cls._generate_qubo_coefficients(
                searchspace.wrap_binary(fitness_function), input_size),
                                   input_size)
