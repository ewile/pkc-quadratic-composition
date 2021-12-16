from typing import List
from copy import deepcopy

from transformations import (
    Transformation,
    AffineTransformation,
    QuadraticComposition,
)
from my_maths import (
    Polynomial,
    get_tuple_index,
    Vector,
    substract,
    eval_polynomial,
)
import specs


class SecretKey:
    """
    dimensions represents [a_1, .. a_m] (see the article)
    """

    def __init__(self, dimensions: List[int]) -> None:
        self.dimensions = dimensions
        self.transformations: List[Transformation] = []
        self.transformations.append(
            AffineTransformation(dimensions[0], dimensions[1])
        )
        for index in range(1, len(dimensions) - 1):
            self.transformations.append(
                QuadraticComposition(dimensions[index])
            )
            self.transformations.append(
                AffineTransformation(dimensions[index], dimensions[index + 1])
            )

    def get_dimensions(self) -> List[int]:
        return self.dimensions

    def get_n_input(self) -> int:
        return self.dimensions[0]

    def get_n_output(self) -> int:
        return self.dimensions[-1]

    def get_n_transformations(self) -> int:
        return len(self.transformations)

    def get_n_params(self) -> int:
        n_params = 0
        for transformation in self.transformations:
            n_params += transformation.get_n_params()
        return n_params

    def get_params(self) -> List[int]:
        params = []
        for transformation in self.transformations:
            params += transformation.get_params()
        return params

    def set_params(self, params: List[int]) -> None:
        param_index = 0
        for transformation in self.transformations:
            n_params_for_transformation = transformation.get_n_params()
            transformation_params = params[
                param_index: param_index + n_params_for_transformation
            ]
            transformation.set_params(transformation_params)
            param_index += n_params_for_transformation

    def __str__(self) -> str:
        s = "SECRET KEY:"
        s += "\n-> " + "p = " + str(specs.p)
        s += "\n-> " + "dimensions: = " + str(self.dimensions)
        for index in range(self.get_n_transformations()):
            s += (
                "\n\ntransformation "
                + str(index)
                + ":\n\n"
                + str(self.transformations[index])
            )
        s += "\n\n"
        return s


class PublicKey:
    """A public key consists in a list of 'n_output' (multivariate) Polynomials.
    They all have 'n_input' variables.
    """

    def __init__(self, n_input: int, n_output: int) -> None:
        self.n_output = n_output
        self.n_input = n_input
        self.polynomials = [Polynomial(n_input) for _ in range(n_output)]

    def get_n_input(self) -> int:
        return self.n_input

    def get_n_output(self) -> int:
        return self.n_output

    def set_polynomial(
        self, index: int, multivariate_polynomial: Polynomial
    ) -> None:
        assert 0 <= index < self.n_output
        self.polynomials[index] = multivariate_polynomial

    def get_polynomial(self, output_index: int) -> Polynomial:
        assert 0 <= output_index < self.n_output
        return self.polynomials[output_index]

    def __str__(self) -> str:

        s = "PUBLIC KEY:"
        s += (
            "\n\n-> "
            + "p = "
            + str(specs.p)
            + " (these multivariate polynomials must be considered modulo p)"
        )
        s += "\n-> " + str(self.n_output) + " lines (ie output variables)"
        s += "\n-> " + str(self.n_input) + " input variables"
        for line_index in range(self.n_output):
            polynomial = self.polynomials[line_index]
            s += (
                "\n\nline number "
                + str(line_index + 1)
                + " :  "
                + str(polynomial)
            )
        s += "\n\n"
        return s


def generate_random_secret_key(dimensions: List[int]) -> SecretKey:
    secret_key = SecretKey(dimensions)
    for transformation in secret_key.transformations:
        transformation.set_random_coefs()
    return secret_key


def get_symbol_coef(symbol_index: int, n_total_symbols: int):
    pol = Polynomial(n_total_symbols)
    pol.set_coef(get_tuple_index(symbol_index, n_total_symbols), 1)
    return pol


def generate_public_key(secret_key: SecretKey, verbose: bool) -> PublicKey:
    if verbose:
        print("generating public key:")
    public_key = PublicKey(secret_key.get_n_input(), secret_key.get_n_output())
    n_input = secret_key.get_n_input()
    vector = Vector(n_input)
    for var_index in range(0, n_input):
        pol_var = Polynomial(n_input)
        pol_var.set_coef(get_tuple_index(var_index, n_input), 1)
        vector[var_index] = pol_var
    for transformation in secret_key.transformations:
        vector = transformation.eval(vector)

    for index in range(len(vector)):
        polynomial = vector[index]
        public_key.set_polynomial(index, polynomial)

    return public_key


def encrypt(message: Vector, public_key: PublicKey) -> Vector:
    """message : a Vector of integers"""
    assert len(message) == public_key.get_n_input()
    encrypted_message = Vector(public_key.get_n_output())
    for line_index in range(public_key.get_n_output()):
        encrypted_message[line_index] = eval_polynomial(
            public_key.get_polynomial(line_index), message
        )
    return encrypted_message


def decrypt(encrypted_message: Vector, secret_key: SecretKey):
    """Returns a set containing all the possible decrypted messages
    (tuple of int) correspnding to encrypted_message (tuple of int).
    """
    assert len(encrypted_message) == secret_key.get_n_output()

    solutions = [encrypted_message]

    for transformation in secret_key.transformations[::-1]:
        copy_solutions = deepcopy(solutions)
        solutions = []
        for sol in copy_solutions:
            solutions += transformation.inverse(sol)

    return solutions


def generate_symbolic_secret_key(dimensions: List[int]) -> SecretKey:
    secret_key = SecretKey(dimensions)
    n_params = secret_key.get_n_params()
    symbol_index = 0
    for transformation in secret_key.transformations:
        transformation_n_params = transformation.get_n_params()
        transformation_params = [None] * transformation_n_params
        for param_index in range(0, transformation_n_params):
            transformation_params[param_index] = get_symbol_coef(
                symbol_index, n_params
            )
            symbol_index += 1
        transformation.set_params(transformation_params)
    return secret_key


def generate_system_needed_to_break_secret_key_security(
    dimensions,
) -> List[Polynomial]:
    """Recovering the public key from the secret key is equivalent to
    solving a system of multivariate polynomials. Returns a example of such
    a system
    """

    int_secret_key = generate_random_secret_key(dimensions)
    int_public_key = generate_public_key(int_secret_key, verbose=False)

    symbolic_secret_key = generate_symbolic_secret_key(dimensions)

    symbolic_public_key = generate_public_key(
        symbolic_secret_key, verbose=False
    )
    print(symbolic_public_key)

    system_needed_to_break_secret_key_security = []
    for polynomial_index in range(dimensions[-1]):
        for coef_index in symbolic_public_key.get_polynomial(
            polynomial_index
        ).coefs:
            int_value = int_public_key.get_polynomial(
                polynomial_index
            ).get_coef(coef_index)
            polynomial_value = symbolic_public_key.get_polynomial(
                polynomial_index
            ).get_coef(coef_index)
            line = substract(polynomial_value, int_value)
            system_needed_to_break_secret_key_security.append(line)
    return system_needed_to_break_secret_key_security
