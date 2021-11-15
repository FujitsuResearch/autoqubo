from autoqubo.binarization import Binarization
import unittest


class TestBinarizationMethods(unittest.TestCase):

    def test_decode(self):
        self.assertEqual(Binarization.uint.decode([1, 0, 0]), 1)
        self.assertEqual(Binarization.uint.decode([1, 1, 0]), 3)
        self.assertEqual(Binarization.uint.decode([1, 1, 1]), 7)

    def test_encode(self):
        self.assertEqual(Binarization.uint.encode(1, 3), [1, 0, 0])
        self.assertEqual(Binarization.uint.encode(2, 3), [0, 1, 0])
        self.assertEqual(Binarization.uint.encode(7, 3), [1, 1, 1])


if __name__ == '__main__':
    unittest.main()
