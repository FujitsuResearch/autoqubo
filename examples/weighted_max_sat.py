def f(x):
    x1, x2, x3 = x
    val = 0

    # clause 1
    if x1 or not x2:
        val += 3
    # clause 2
    if x3:
        val += 1
    # clause 3
    if not x3 or x2:
        val += 4

    return val


if __name__ == '__main__':
    from itertools import product
    from autoqubo import SamplingCompiler, Utils

    print("Sampling:")
    for x in product(range(2), repeat=3):
        print(f"x={x}, f(x)={f(x)}")

    qubo, offset = SamplingCompiler.generate_qubo_matrix(f, 3)
    if SamplingCompiler.test_qubo_matrix(f, qubo, offset):
        print("QUBO generation succesful")
    else:
        print("QUBO generation failed - the objective function is not quadratic")

    print("QUBO matrix:")
    print(qubo)
    print("QUBO coefficients")
    print(f"x[] = {offset}")
    for key, coefficient in Utils.get_matrix_dict_repr(qubo).items():
        i, j = key
        if i == j:
            print(f"x[{i+1}] = {coefficient}")
        else:
            print(f"x[{i+1}, {j+1}] = {coefficient}")

    print("Best solutions (minimize)")
    solutions, energy_values = Utils.solve(qubo, offset)
    for s, e in zip(solutions, energy_values):
        print(f"x={s}, energy={e}")

    print("Best solutions (maximize)")
    solutions, energy_values = Utils.solve(-qubo, -offset)
    for s, e in zip(solutions, energy_values):
        print(f"x={s}, energy={-e}")
