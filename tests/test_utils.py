from autoqubo.utils import Utils
import unittest
import numpy as np


class TestUtilsMethods(unittest.TestCase):

    def test_training_set(self):
        self.assertEqual(Utils.energy(np.array([[1, 3], [0, 4]]), np.array([1, 1])), 8)


if __name__ == '__main__':
    unittest.main()
