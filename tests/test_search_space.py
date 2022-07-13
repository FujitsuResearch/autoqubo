from autoqubo.search_space import SearchSpace
from autoqubo.binarization import Binarization
import unittest


class TestSearchSpaceMethods(unittest.TestCase):

    def test_search_space(self):
        s = SearchSpace([('a', Binarization.uint, 3), ('b', Binarization.uint, 3)])
        a, b = s.decode([1, 1, 0, 0, 1, 1])
        self.assertEqual(a, 3)
        self.assertEqual(b, 6)
        self.assertEqual(s.size, 6)


if __name__ == '__main__':
    unittest.main()
