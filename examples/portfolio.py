import csv
import numpy as np


with open('data/portfolio.txt') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    n, budget = map(int, next(csv_reader))
    cov_matrix = np.zeros((n, n))
    mean_vector = np.zeros(n)
    for row in csv_reader:
        if len(row) == 2:
            i, v = map(int, row)
            mean_vector[i] = v
        elif len(row) == 3:
            i, j, v = map(int, row)
            cov_matrix[i, j] = v
        else:
            raise ValueError


print("means:", mean_vector)
print("cov:")
print(cov_matrix)


def variance(x):
    """
    Variance
    """
    return x@cov_matrix@x


def mean(x):
    """
    Mean return
    """
    return x@mean_vector


def constraint(x):
    """
    Budget constraint
    """
    return (x.sum() - budget)**2


A, B, C = 1, -1, 100


def f(x):
    """
    Mean-variance portfolio optimization model
    """
    return A*variance(x) + B*mean(x) + C*constraint(x)


if __name__ == '__main__':
    from autoqubo import Binarization, SamplingCompiler, SearchSpace, Utils

    s = SearchSpace()
    weights_vector = Binarization.get_uint_vector_type(3, 3)
    s.add('x', weights_vector, 3 * 3)

    qubo, offset = SamplingCompiler.generate_qubo_matrix(f, s.size, s)
    if SamplingCompiler.test_qubo_matrix(f, qubo, offset):
        print("QUBO generation successful")
    else:
        print("QUBO generation failed - the objective function is not quadratic")

    print("QUBO matrix:")
    print(qubo)
    print("QUBO offset")
    print(f"x[] = {offset}")

    print("Best solutions (minimize)")
    solutions, energy_values = Utils.solve(qubo, offset)
    for sol, e in zip(solutions, energy_values):
        x = s.decode_dict(sol)['x']
        print(
            f"x={x}, "
            f"energy={e}, "
            f"obj={variance(x)-mean(x)}, "
            f"constraint={constraint(x)}"
        )
