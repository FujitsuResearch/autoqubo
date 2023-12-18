import numpy as np
import sys
import warnings
from multiprocessing import Pool
from typing import Callable, Optional, Tuple
from typing_extensions import Literal
from autoqubo.penalty_weights import generate_penalty


class SamplingCompiler:
    """
    Provides .generate_qubo_matrix() method that allows to transform a function into a QUBO model.
    """

    @staticmethod
    def _indices_iterator(input_size):
        yield tuple()

        for i in range(input_size):
            yield (i,)

        for i in range(input_size):
            for j in range(i + 1, input_size):
                yield (i, j)

    @staticmethod
    def _new_training_sample(input_size, idx):
        sample = [0] * input_size
        for i in idx:
            sample[i] = 1
        return sample

    @staticmethod
    def _new_test_sample(input_size):
        sample = tuple(np.random.randint(2, size=(input_size,)))
        return sample

    @staticmethod
    def _get_training_samples(input_size):
        return (
            SamplingCompiler._new_training_sample(input_size, idx)
            for idx in SamplingCompiler._indices_iterator(input_size)
        )

    @staticmethod
    def _get_test_samples(input_size, num_test_samples):
        # Compute maximum number of testing samples and adjust
        max_test_samples = 2**input_size - (1 + input_size * (input_size + 1) // 2)
        if num_test_samples > max_test_samples:
            print(
                f"*** Warning, requested test size is {num_test_samples}, which is larger than the maximum of {max_test_samples}"
            )
            num_test_samples = max_test_samples

        # Compute the test samples
        test_samples = set()
        while len(test_samples) < num_test_samples:
            sample = SamplingCompiler._new_test_sample(input_size)
            # We know that the training set contains 0-hot, 1-hot and 2-hot examples.
            # Hence all samples with at least 3 ones cannot be in the training set
            if sum(sample) > 2:
                test_samples.add(sample)

        return test_samples

    @staticmethod
    def _generate_training_output(fitness_function, input_size, use_multiprocessing=True):
        """
        Gather and evaluate fitness function for training samples.

        :param fitness_function: callable
            The fitness function to be evaluated.
        :param input_size: int
            The size of the input for the fitness function.
        :param use_multiprocessing: bool, optional
            Flag to enable/disable multiprocessing for generating training output.
        :return: list
            List containing the fitness values for each training sample.
        """
        if use_multiprocessing is False:
            results = (
                fitness_function(sample)
                for sample in SamplingCompiler._get_training_samples(input_size)
            )
        else:
            if "ipykernel" in sys.modules:
                msg = "Multiprocessing is enabled by default, but not available in interactive sessions such as jupyter notebooks. Sampling is done without multiprocessing."
                warnings.warn(msg, UserWarning)
                return SamplingCompiler._generate_training_output(
                    fitness_function, input_size, use_multiprocessing=False
                )
            samples = list(SamplingCompiler._get_training_samples(input_size))
            with Pool() as pool:
                results = pool.map(fitness_function, samples)
        return results

    @classmethod
    def _generate_qubo_coefficients(
        cls, fitness_function, input_size, use_multiprocessing = True
    ):
        coefficients = []
        for output, index in zip(
            cls._generate_training_output(
                fitness_function, input_size, use_multiprocessing
            ),
            cls._indices_iterator(input_size),
        ):
            coefficients.append(
                output
                - (0 if len(index) < 2 else sum(coefficients[i + 1] for i in index))
                - (0 if index == tuple() else coefficients[0])
            )
        return coefficients

    @staticmethod
    def _qubo_matrix(coefficients, input_size):
        # use general purpose object dtype first
        qubo = np.zeros((input_size, input_size), dtype="object")
        k = 0
        for idx in SamplingCompiler._indices_iterator(input_size):
            if len(idx) == 1:
                qubo[idx[0], idx[0]] = coefficients[k]
            if len(idx) == 2:
                qubo[idx[0], idx[1]] = coefficients[k]
            k += 1
        try:  # try to convert to float array
            qubo = qubo.astype(np.float64)
        except TypeError as e:
            # keep object dtype if qubo is symbolic
            if "symbol" in str(e) or "expression" in str(e):
                pass
            else:
                raise
        return qubo, coefficients[0]

    @classmethod
    def generate_qubo_matrix(
        cls,
        fitness_function: Callable,
        input_size: int,
        use_multiprocessing: bool = True,
        searchspace: Optional["SearchSpace"] = None,
    ) -> Tuple[np.array, int]:
        """
        Generates a QUBO matrix for a given function.
        :param fitness_function: Callable
            Function to be compiled.
        :param input_size: int
            number of binary variables in the function input.
        :param use_multiprocessing: bool, optional
            Flag to enable/disable multiprocessing for generating training output.
        :param searchspace: SearchSpace
            Optional parameter describing the arguments of the function.
        :return: Q, c
            Q: QUBO matrix
            c: offset / constant term
        """
        if searchspace is None:
            return cls._qubo_matrix(
                cls._generate_qubo_coefficients(
                    fitness_function, input_size, use_multiprocessing
                ),
                input_size,
            )
        else:
            return cls._qubo_matrix(
                cls._generate_qubo_coefficients(
                    searchspace.wrap_binary(fitness_function),
                    input_size,
                    use_multiprocessing,
                ),
                input_size,
            )

    @classmethod
    def test_qubo_matrix(
        cls,
        fitness_function: Callable,
        qubo_matrix: np.array,
        offset: float,
        search_space: Optional["SearchSpace"] = None,
        num_test_samples: int = -1,
        epsilon: float = 1e-8,
    ) -> bool:
        """
        Performs a test to see whether the qubification process was successful.
        The process is not successful if the function is not quadratic
        :param fitness_function: Callable
            Function to be compiled.
        :param qubo_matrix: np.array
            The QUBO being tested
        :param offset: float
            The constant term
        :param search_space: Optional['SearchSpace']
            Optional parameter describing the arguments of the function.
        :param num_test_samples: int
            number of test points to use to test the correctness of the QUBO.
            If set to -1, will use n testing point
        :param epsilon: float
            precision of comparison between function value and qubo value
        :return: bool
            True if the test succeeded (meaning function is quadratic, False if it failed(
        """

        if search_space is None:
            binary_func = fitness_function
        else:
            binary_func = search_space.wrap_binary(fitness_function)

        input_size = qubo_matrix.shape[0]
        num_test_samples = input_size if num_test_samples < 0 else num_test_samples
        test_samples = cls._get_test_samples(input_size, num_test_samples)

        for sample in test_samples:
            target = binary_func(sample)
            actual = sample @ qubo_matrix @ sample + offset
            if abs(actual - target) > epsilon:
                return False

        return True

    @classmethod
    def generate_qubo(
        cls,
        cost: Callable,
        constraints: Callable,
        input_size: int,
        penalty_method: Optional[Literal["sum", "pnform", "verma_lewis"]] = "sum",
        penalty_weight: Optional[float] = None,
        use_multiprocessing: bool = False,
        searchspace: Optional["SearchSpace"] = None,
    ) -> Tuple[np.array, int]:
        """
        Generates a combined QUBO matrix for given cost and constraints.
        :param cost: Callable
            cost function
        :param constraints: Callable
            all problem constraints in one function
        :param input_size: int
            number of binary variables in the function input.
        :param penalty_method: Literal
            optional choice of how to generate penalty weight
        :param penalty_weight: int
            optional custom penalty weights
        :param use_multiprocessing: bool, optional
            Flag to enable/disable multiprocessing for generating training output.
        :param searchspace: SearchSpace
            Optional parameter describing the arguments of the function.
        :return: Q, c
            Q: QUBO matrixp cost
            c: offset / constant term
        """
        cost_qubo, cost_offset = cls.generate_qubo_matrix(
            cost, input_size, use_multiprocessing, searchspace
        )
        constraint_qubo, constraint_offset = cls.generate_qubo_matrix(
            constraints, input_size, use_multiprocessing, searchspace
        )
        # only generate penalty weight if none is given
        if not penalty_weight:
            penalty_weight = generate_penalty(
                penalty_method, cost_qubo, constraint_qubo
            )
        Q = cost_qubo + penalty_weight * constraint_qubo
        offset = cost_offset + constraint_offset
        return Q, offset
