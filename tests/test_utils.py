from autoqubo.utils import Utils
import unittest
import numpy as np


class TestUtilsMethods(unittest.TestCase):

    def test_training_set(self):
        self.assertEqual(Utils.energy(np.array([[1, 3], [0, 4]]), np.array([1, 1])), 8)

    def test_get_matrix_dict_repr(self):
        self.assertEqual(
            Utils.get_matrix_dict_repr(np.array([[0, 1], [0, 2]])),
            {(0, 1): 1, (1, 1): 2}
        )

    def test_get_solution_vector_repr(self):
        self.assertEqual(
            Utils.get_solution_vector_repr({0: 1, 1: 4, 2: 7}),
            [1, 4, 7]
        )


if __name__ == '__main__':
    unittest.main()
