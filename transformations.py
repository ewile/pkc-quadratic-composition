from random import randint
import abc
from typing import List
from copy import deepcopy

import specs
from my_maths import (
    Vector,
    Polynomial,
    get_tuple_index,
    increase_coef_by,
    solve_linear_system,
    eval_polynomial,
    multiply,
    solve_quadratic_polynomial,
)


class Transformation:
    """A 'Transformation' represents a function with multiple inputs and multiple
    outputs"""

    def __init__(self, n_input: int, n_output: int) -> None:
        self.n_input = n_input
        self.n_output = n_output

    @abc.abstractmethod
    def get_n_params(self) -> int:
        return

    @abc.abstractmethod
    def get_params(self) -> List[int]:
        return

    @abc.abstractmethod
    def set_params(self, params: List[int]) -> None:
        return

    def set_random_coefs(self) -> None:
        params = [randint(0, specs.p - 1) for _ in range(self.get_n_params())]
        self.set_params(params)

    @abc.abstractmethod
    def eval(self, input: Vector) -> Vector:
        return

    @abc.abstractmethod
    def inverse(self, output: Vector) -> List[Vector]:
        """Returns the list of the vectors X such that transformation(X) = output
        (Returns [] if there is no solutions)"""
        return

    @abc.abstractmethod
    def __str__(self) -> str:
        return

    def __repr__(self):
        return str(self)


class AffineTransformation(Transformation):
    """An affine transformation modulo p.
    That is to say: T(X) = A.X + B
    where A is a fixed matrix, and B a fixed vector."""

    def __init__(self, n_input: int, n_output: int) -> None:
        Transformation.__init__(self, n_input, n_output)
        self.matrix = [Polynomial(n_input) for _ in range(n_output)]

    @abc.abstractmethod
    def get_n_params(self) -> int:
        return self.n_output * (self.n_input + 1)

    @abc.abstractmethod
    def get_params(self) -> List[int]:
        params = [0] * self.get_n_params()
        param_index = 0
        n_variables = self.matrix[0].get_number_of_variables()
        for linear_combination in self.matrix:
            for var_index in range(-1, n_variables):
                params[param_index] = linear_combination.get_coef(
                    get_tuple_index(var_index, n_variables)
                )
                param_index += 1
        return params

    @abc.abstractmethod
    def set_params(self, params: List[int]) -> None:
        n_variables = self.matrix[0].get_number_of_variables()
        param_index = 0
        for linear_combination in self.matrix:
            for var_index in range(-1, n_variables):
                linear_combination.set_coef(
                    get_tuple_index(var_index, n_variables),
                    params[param_index],
                )
                param_index += 1

    def eval(self, input: Vector) -> Vector:
        output = Vector(self.n_output)
        for index in range(self.n_output):
            output[index] = eval_polynomial((self.matrix[index]), input)
        return output

    def inverse(self, output: Vector) -> List[Vector]:
        """Linear system solver with Gaussian elimination:"""
        assert self.n_output == len(output)

        new_linear_system = []
        for index, linear_combination in enumerate(self.matrix):
            new_linear_combination = deepcopy(linear_combination)
            tuple_index = get_tuple_index(
                -1, linear_combination.get_number_of_variables()
            )
            increase_coef_by(
                new_linear_combination,
                tuple_index,
                multiply(-1, output[index]),
            )
            new_linear_system.append(new_linear_combination)
        return solve_linear_system(new_linear_system)

    def __str__(self) -> str:
        s = (
            "Affine transformation: n_input = "
            + str(self.n_input)
            + ", n_output = "
            + str(self.n_output)
        )
        for linearCombination in self.matrix:
            s += "\n" + str(linearCombination)
        return s


class QuadraticComposition(Transformation):
    """A list of quadratic functions:
    f1(x) = a.x^2 + b.x + c
    f2(x) = d.x^2 + e.x + f
    f3(x) = g.x^2 + h.x + i
    (n_input = 3 in this example)

    input: vector -> [x, y, z]
    output: vector -> [f1(x), f2(y), f3(z)]

    Reprents the functions named 'Q' in the article

    """

    def __init__(self, n_input) -> None:
        Transformation.__init__(self, n_input, n_input)
        self.polynomials = [Polynomial(1) for _ in range(n_input)]

    @abc.abstractmethod
    def get_n_params(self) -> int:
        return 3 * self.n_input

    @abc.abstractmethod
    def get_params(self) -> List[int]:
        params = [0] * self.get_n_params()
        param_index = 0
        for quadratic in self.polynomials:
            for power_index in range(0, 3):
                params[param_index] = quadratic.get_coef((power_index,))
                param_index += 1
        return params

    @abc.abstractmethod
    def set_params(self, params: List[int]) -> None:
        param_index = 0
        for quadratic in self.polynomials:
            for power_index in range(0, 3):
                quadratic.set_coef((power_index,), params[param_index])
                param_index += 1

    def eval(self, input: Vector) -> Vector:
        output = Vector(self.n_input)
        for index in range(self.n_input):
            single_variable_input = Vector(1)
            single_variable_input[0] = input[index]
            output[index] = eval_polynomial(
                self.polynomials[index], single_variable_input
            )
        return output

    def inverse(self, output: Vector) -> List[Vector]:
        """returns the lis of the vectors X such that
        quadratic_composition(X) = output"""
        assert len(self.polynomials) == len(output)

        solutions = [[]]

        for index in range(self.n_input):
            quadratic_solutions = solve_quadratic_polynomial(
                self.polynomials[index], output[index]
            )
            if quadratic_solutions == []:
                return []
            copy_solutions = deepcopy(solutions)
            solutions = []
            for sol in copy_solutions:
                for s in quadratic_solutions:
                    solutions.append(sol + [s])
        return solutions

    def __str__(self) -> str:
        s = (
            "quadratic composition: n_input = "
            + str(self.n_input)
            + ", n_output = "
            + str(self.n_output)
        )
        for polynomial in self.polynomials:
            s += "\n" + str(polynomial)
        return s
