from autoqubo.binarization import Binarization
import unittest
import numpy as np


class TestBinarizationMethods(unittest.TestCase):

    def test_decode(self):
        self.assertEqual(Binarization.uint.decode([1, 0, 0]), 1)
        self.assertEqual(Binarization.uint.decode([1, 1, 0]), 3)
        self.assertEqual(Binarization.uint.decode([1, 1, 1]), 7)

    def test_encode(self):
        self.assertEqual(Binarization.uint.encode(1, 3), [1, 0, 0])
        self.assertEqual(Binarization.uint.encode(2, 3), [0, 1, 0])
        self.assertEqual(Binarization.uint.encode(7, 3), [1, 1, 1])

    def test_vector(self):
        testing_type = Binarization.get_uint_vector_type(3, 4)
        self.assertTrue((testing_type.decode(
            Binarization.uint.encode(1, 3) +
            Binarization.uint.encode(2, 3) +
            Binarization.uint.encode(3, 3) +
            Binarization.uint.encode(4, 3)
        ) == np.array((1, 2, 3, 4))).all())

        self.assertEqual(
            testing_type.encode([1, 2, 3, 4], 12),
            Binarization.uint.encode(1, 3) +
            Binarization.uint.encode(2, 3) +
            Binarization.uint.encode(3, 3) +
            Binarization.uint.encode(4, 3)
        )


if __name__ == '__main__':
    unittest.main()
