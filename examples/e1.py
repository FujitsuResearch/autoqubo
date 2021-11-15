from autoqubo.binarization import Binarization
from autoqubo.sampling_compiler import SamplingCompiler
from autoqubo.search_space import SearchSpace
from autoqubo.utils import Utils


s = SearchSpace()
s.add('a', Binarization.uint, 3)
s.add('b', Binarization.uint, 3)


def ff(a, b):
    return 2 * a + 3 * b


q, offset = SamplingCompiler.generate_qubo_matrix(ff, s.size, s)

print(q)

x = s.encode({'a': 3, 'b': 6})

print(Utils.energy(q, x, offset=offset))
print(ff(3, 6))
