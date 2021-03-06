from itertools import combinations

# Number of vertices of the graph
n = 10

# List of edges of the graph
E = {
    (1, 3),
    (2, 4),
    (2, 5),
    (3, 6),
    (3, 9),
    (3, 7),
    (3, 4),
    (4, 6),
    (4, 9),
    (4, 7),
    (4, 8),
    (5, 7),
    (5, 8),
    (6, 7),
    (6, 9),
    (7, 8),
    (7, 9),
    (7, 10)
}


# Objective function (to maximize)
def objective_function(x):
    return sum(x)  # to maximise


# Logical constraint
def logical_constraint(x):
    for i, j in combinations(range(n), 2):
        if (x[i] == 1 and x[j] == 1) and (i+1, j+1) not in E:
            return False
    return True


def modified_constraint(x):
    sum_violations = 0
    for i, j in combinations(range(n), 2):
        if (x[i] == 1 and x[j] == 1) and (i+1, j+1) not in E:
            sum_violations += 1
    return sum_violations


# Penalty coefficients
A, B = 1, n+1


# Unconstrained formulation (to minimize)
def f_quadratic(x):
    return -A*objective_function(x) + B*int(modified_constraint(x))

def f_ho(x):
    return -A*objective_function(x) + B*int(logical_constraint(x))


if __name__ == '__main__':
    from itertools import product
    from autoqubo import SamplingCompiler, Utils


    # Conversion fails if f is higher order
    qubo, offset = SamplingCompiler.generate_qubo_matrix(f_ho, n)
    if SamplingCompiler.test_qubo_matrix(f_ho, qubo, offset):
        print("QUBO generation successful")
    else:
        print("QUBO generation failed - the objective function is not quadratic")

    # Conversion works if using the smoothed, modified constrained
    qubo, offset = SamplingCompiler.generate_qubo_matrix(f_quadratic, n)
    if SamplingCompiler.test_qubo_matrix(f_quadratic, qubo, offset):
        print("QUBO generation successful")
    else:
        print("QUBO generation failed - the objective function is not quadratic")

    print("QUBO matrix:")
    print(qubo)
    print("QUBO coefficients")
    print(f"x[] = {offset}")
    for key, coefficient in Utils.get_matrix_dict_repr(qubo).items():
        i, j = key
        if i == j:
            print(f"x[{i}] = {coefficient}")
    for key, coefficient in Utils.get_matrix_dict_repr(qubo).items():
        i, j = key
        if i != j:
            print(f"x[{i}, {j}] = {coefficient}")

    print("Best solutions (minimize)")
    solutions, energy_values = Utils.solve(qubo, offset)
    for s, e in zip(solutions, energy_values):
        print(f"x={s}, energy={e}, obj={objective_function(s)}, constraint={modified_constraint(s)}")
