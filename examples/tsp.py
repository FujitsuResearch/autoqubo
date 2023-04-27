import numpy as np
from autoqubo import SamplingCompiler
from autoqubo.symbolic import symbolic_matrix, insert_values


def constraint(x):
    """2-way one hot"""
    n = int(np.sqrt(len(x)))
    x = np.array(x).reshape(n, n)
    row_sum = x.sum(axis=1)
    col_sum = x.sum(axis=0)
    row_violations = np.absolute(1 - row_sum) ** 2
    col_violations = np.absolute(1 - col_sum) ** 2
    return row_violations.sum() + col_violations.sum()


def tour_length(x, A, n):
    """tsp tour length given adjacency matrix A"""
    tour_length = 0
    for k in range(n):  # step k
        for i in range(n):  # city i
            for j in range(n):  # city j
                # weight of edge (i,j) at step k, k+1
                city_i = k * n + i
                city_j = (((k + 1) % n) * n) + j
                tour_length += A[i, j] * x[city_i] * x[city_j]
    return tour_length


if __name__ == "__main__":
    n = 4
    # explicit sampling
    A = np.array([[0, 7, 4, 8], [7, 0, 15, 11], [4, 15, 0, 2], [8, 11, 2, 0]])
    cost = lambda x: tour_length(x, A, len(A))
    qubo, offset = SamplingCompiler.generate_qubo(cost, constraint, input_size=n**2)

    # symbolic sampling
    sym_mat = symbolic_matrix(n, n, positive=True)
    cost = lambda x: tour_length(x, sym_mat, n)

    # generate symbolic QUBO
    sym_qubo, offset = SamplingCompiler.generate_qubo(cost, constraint, n**2)
    # insert explicit values from single instance
    qubo2 = insert_values(sym_qubo, A)

    print("Explicit Sampling returns same matrix as symbolic sampling:")
    print((qubo == qubo2).all())
