import numpy as np


class SamplingCompiler:

    @staticmethod
    def indices_iterator(input_size):
        yield tuple()

        for i in range(input_size):
            yield (i,)

        for i in range(input_size):
            for j in range(i+1, input_size):
                yield (i, j)

    @staticmethod
    def new_training_sample(input_size, idx):
        sample = [0] * input_size
        for i in idx:
            sample[i] = 1
        return sample

    @staticmethod
    def get_training_samples(input_size):
        return (SamplingCompiler.new_training_sample(input_size, idx)
                for idx in SamplingCompiler.indices_iterator(input_size))

    @staticmethod
    def generate_training_output(fitness_function, input_size):
        return (fitness_function(sample) for sample in SamplingCompiler.get_training_samples(input_size))

    @classmethod
    def generate_qubo_coefficients(cls, fitness_function, input_size):
        coefficients = []
        for output, index in zip(cls.generate_training_output(fitness_function, input_size),
                                 cls.indices_iterator(input_size)):
            coefficients.append(output - (0 if len(index) < 2 else sum(coefficients[i+1] for i in index)) -
                                (0 if index == tuple() else coefficients[0]))
        return coefficients

    @staticmethod
    def qubo_matrix(coefficients, input_size, dtype=np.float64):
        qubo = np.zeros((input_size, input_size), dtype=dtype)
        k = 0
        for idx in SamplingCompiler.indices_iterator(input_size):
            if len(idx) == 1:
                qubo[idx[0], idx[0]] = coefficients[k]
            if len(idx) == 2:
                qubo[idx[0], idx[1]] = coefficients[k]
            k += 1
        return qubo, coefficients[0]

    @classmethod
    def generate_qubo_matrix(cls, fitness_function, input_size, searchspace=None):
        if searchspace is None:
            return cls.qubo_matrix(cls.generate_qubo_coefficients(fitness_function, input_size), input_size)
        else:
            return cls.qubo_matrix(cls.generate_qubo_coefficients(
                searchspace.wrap_binary(fitness_function), input_size),
                                   input_size)


if __name__ == '__main__':

    def f(x):
        return x[0]

    def g(x):
        return 1 + 2 * x[0] + 3 * x[1] + 4 * x[0] * x[1]

    sc = SamplingCompiler()
    c = SamplingCompiler.get_training_samples(1)
    print(c)
    print(list(c))
    c = SamplingCompiler.generate_training_output(f, 1)
    print(c)
    print(list(c))
    c = SamplingCompiler.generate_qubo_coefficients(f, 1)
    print(c)
    c = SamplingCompiler.generate_qubo_coefficients(g, 2)
    print(c)
