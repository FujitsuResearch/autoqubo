"""
provides functions around symbolic variables
and substitution thereof. symbols are designed
such that sampling works fine out-of-the-box
as if we're using proper numeric values
"""

from sympy import symbols
import numpy as np


def symbolic_matrix(n_rows, m_cols, positive=False):
    """creates a symbolic matrix of vars of given size
    Args:
        n_rows(int): number of rows
        m_cols(int): number of columns
        positive(bool): if variables can be assumed to be positive
    Returns:
        np.ndarray[sympy.core.symbol.Symbol]: matrix of symbolic vars
    """
    symbolic_array = []
    for row in range(n_rows):
        # indices are seperated by whitespace, e.g. "s0 2"
        row = symbols([f"s{row}\ {j}" for j in range(m_cols)], positive=positive)
        symbolic_array.append(row)
    symbolic_array = np.array(symbolic_array)
    return symbolic_array


def formula_wise_substitution(expression, source_matrix):
    """inserts numeric values from source matrix into
    all symbolic variables present in sympy expression
    Args:
        expression(sympy.core.add.Add): formula with variables
        source_matrix(np.ndarray): entry at i,j gives value for si j
    Returns:
        result(float): value after evaluation of the formula
    """
    try:
        free_vars = list(expression.free_symbols)
        sub_dir = {}
        for var in free_vars:
            # assume var indices are seperated by whitespace
            i, j = [int(k) for k in var.name[1:].split(" ")]
            sub_dir[var] = source_matrix[i, j]
        result = expression.subs(sub_dir)
        return result
    # don't do anything if there are no symbols
    except AttributeError:
        return expression


# vectorize function so we can apply it to qubo matrix
def insert_values(qubo, source_matrix):
    vec_func = np.vectorize(lambda x: formula_wise_substitution(x, source_matrix))
    return vec_func(qubo)
