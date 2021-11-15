from autoqubo.sampling_compiler import SamplingCompiler
import unittest
import numpy as np


def f(x):
    return x[0]


def g(x):
    return 1 + 2*x[0] + 3*x[1] + 4*x[0]*x[1]


class TestSamplingCompilerMethods(unittest.TestCase):

    def test_training_set(self):
        self.assertEqual(list(SamplingCompiler.indices_iterator(1)), [tuple(), (0,)])
        self.assertEqual(list(SamplingCompiler.indices_iterator(2)), [tuple(), (0,), (1,), (0, 1)])

        self.assertEqual(list(SamplingCompiler.get_training_samples(1)), [[0], [1]])
        self.assertEqual(list(SamplingCompiler.get_training_samples(2)), [[0, 0], [1, 0], [0, 1], [1, 1]])

        self.assertEqual(list(SamplingCompiler.generate_training_output(f, 1)), [0, 1])
        self.assertEqual(list(SamplingCompiler.generate_training_output(g, 2)), [1, 3, 4, 10])

        self.assertEqual(list(SamplingCompiler.generate_qubo_coefficients(f, 1)), [0, 1])
        self.assertEqual(list(SamplingCompiler.generate_qubo_coefficients(g, 2)), [1, 2, 3, 4])
        # self.assertEqual(list(SamplingCompiler.generate_qubo_coefficients(f, 1)), [[0, 0], [1, 0], [0, 1], [1, 1]])

        self.assertTrue(
            (SamplingCompiler.qubo_matrix([1, 2, 3, 4], 2)[0] == np.array([[2, 4], [0, 3]])).all()
        )
        self.assertTrue(
            (SamplingCompiler.generate_qubo_matrix(g, 2)[0] == np.array([[2, 4], [0, 3]])).all()
        )


if __name__ == '__main__':

    unittest.main()
