from autoqubo.sampling_compiler import SamplingCompiler
import unittest
import numpy as np


def f(x):
    return x[0]


def g(x):
    return 1 + 2*x[0] + 3*x[1] + 4*x[0]*x[1]

def h(x):
    return 1 + 3*x[1] + 1*x[0]*x[1] + 2*x[0]*x[2] + 12*x[1]*x[2]

def hc(x):
    return 1 + 3*x[1] + 1*x[0]*x[1]*x[2]

class TestSamplingCompilerMethods(unittest.TestCase):

    def test_training_set(self):
        self.assertEqual(list(SamplingCompiler._indices_iterator(1)), [tuple(), (0,)])
        self.assertEqual(list(SamplingCompiler._indices_iterator(2)), [tuple(), (0,), (1,), (0, 1)])

        self.assertEqual(list(SamplingCompiler._get_training_samples(1)), [[0], [1]])
        self.assertEqual(list(SamplingCompiler._get_training_samples(2)), [[0, 0], [1, 0], [0, 1], [1, 1]])

        self.assertEqual(list(SamplingCompiler._generate_training_output(f, 1)), [0, 1])
        self.assertEqual(list(SamplingCompiler._generate_training_output(g, 2)), [1, 3, 4, 10])

        self.assertEqual(list(SamplingCompiler._generate_qubo_coefficients(f, 1)), [0, 1])
        self.assertEqual(list(SamplingCompiler._generate_qubo_coefficients(g, 2)), [1, 2, 3, 4])
        # self.assertEqual(list(SamplingCompiler.generate_qubo_coefficients(f, 1)), [[0, 0], [1, 0], [0, 1], [1, 1]])

        self.assertTrue(
            (SamplingCompiler._qubo_matrix([1, 2, 3, 4], 2)[0] == np.array([[2, 4], [0, 3]])).all()
        )
        self.assertTrue(
            (SamplingCompiler.generate_qubo_matrix(g, 2)[0] == np.array([[2, 4], [0, 3]])).all()
        )

    def test_test_set(self):
        input_size = 3
        num_test_samples = 2

        test_samples = list(SamplingCompiler._get_test_samples(input_size, num_test_samples))
        self.assertEqual(len(test_samples), 1)
        self.assertEqual(test_samples, [(1,1,1)])

    def test_test_qubo(self):

        # This should succeed because h is quadratic
        qubo, offset = SamplingCompiler.generate_qubo_matrix(fitness_function=h, input_size=3)
        self.assertTrue(SamplingCompiler.test_qubo_matrix(fitness_function=h, qubo_matrix=qubo, offset=offset))

        # This should fail because hc is cubic
        qubo, offset = SamplingCompiler.generate_qubo_matrix(fitness_function=hc, input_size=3)
        self.assertFalse(SamplingCompiler.test_qubo_matrix(fitness_function=hc, qubo_matrix=qubo, offset=offset))

if __name__ == '__main__':

    unittest.main()
