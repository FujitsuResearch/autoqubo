"""
provides functions to generate penalty weights from
QUBO matrices
"""

import numpy as np
import sympy
from typing_extensions import Literal
from typing import Union


def sum_penalty(cost_qubo: np.ndarray) -> Union[float, sympy.core.add.Add]:
    """
    maximimum difference in positive/negative rowsums
    :param cost_qubo: np.ndarray
        qubo matrix only representing the cost function
    :return:
        weight: float or sympy expression
    """
    pos_sum = cost_qubo[cost_qubo > 0].sum()
    neg_sum = cost_qubo[cost_qubo < 0].sum()
    return pos_sum - neg_sum


def pos_neg_penalty(C: np.ndarray) -> float:
    """
    use constants from posi-and negaform represenation of the cost
    following [Boros, Endre & Hammer, Peter & Tavares, Gabriel. (2006).
    Preprocessing of unconstrained quadratic binary optimization]
    :param C: np.ndarray
        cost qubo matrix only representing the cost function
    :return:
        weight: float
    """
    # TODO support symbolic cost matrices
    n = C.shape[0]
    # positive form
    # c_j' = c_j + \sum_{c_ij<0} c_{ij}
    pos_cj = np.array([C[j][j] + C[:j, j][C[:j, j] < 0].sum() for j in range(n)])
    # c_0' = c_0 + \sum_{c_j'<0} c_j'
    pos_c0 = pos_cj[pos_cj < 0].sum()
    # negative form
    neg_cj = np.array([C[j][j] + C[:j, j][C[:j, j] > 0].sum() for j in range(n)])
    neg_c0 = neg_cj[neg_cj > 0].sum()
    return float(neg_c0 - pos_c0)


def verma_lewis(C: np.ndarray) -> float:
    """
    maximum sum of positive/negative row entries
    linear terms are always added, just with different sign
    :param C: np.ndarray
        cost qubo matrix only representing the cost function
    :return:
        weight: float
    """
    # TODO support symbolic cost matrices
    n = C.shape[0]
    # positive entries: c_ii + \sum c_ij (c_ij>0)
    pos_sum = np.array(
        [C[i][i] + C[i, i + 1 :][C[i, i + 1 :] > 0].sum() for i in range(n)]
    )
    # negative entries: -c_ii - \sum c_ij (c_ij<0)
    neg_sum = np.array(
        [-C[i][i] - C[i, i + 1 :][C[i, i + 1 :] < 0].sum() for i in range(n)]
    )
    return float(np.max([pos_sum, neg_sum]))


def generate_penalty(
    penalty_method: Literal["sum", "pnform", "verma_lewis"],
    cost_qubo: np.ndarray,
    constraint_qubo: np.ndarray,
) -> float:
    """
    performs any given penalty method and returns the weight
    :param cost_qubo: np.ndarray
        qubo matrix only representing the cost function
    :param constraint_qubo: np.ndarray
        qubo matrix only representing the constraint functions
    :param penalty_method: Literal
        how to generate penalty weight
    :return:
        weight: float
    """
    if penalty_method == "sum":
        return sum_penalty(cost_qubo)
    elif penalty_method == "pnform":
        return pos_neg_penalty(cost_qubo)
    elif penalty_method == "verma_lewis":
        return verma_lewis(cost_qubo)
    else:
        raise ValueError(f"Unknown penalty method {penalty_method}")
