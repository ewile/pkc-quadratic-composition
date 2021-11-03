from typing import Tuple, List
import random

from math_modulo import solve_linear_system_eq, solve_quadratic_polynomial, Polynomial, compose, get_tuple_index
import specs


class SecretKey:
    """A secret key consists in 'n_lines' lines.
    Each line consists in 'n_compositions' (univariate)
    quadratic polynomials, and a linear combination of 'n_variables' 
    variables.

    The quadratic polynomials are stored in self.quadratic_polynomials, a 2d list:
    self.quadratic_polynomials[line_index][composition_index] :

    During the generation of the public key:
    self.quadratic_polynomials[line_index][0] is composed with a linear combination
    then self.quadratic_polynomials[line_index][1] is composed with the result above
    then self.quadratic_polynomials[line_index][2] and so on ...

    The linear combinations are stored in self.linear_combinations : a list of
    'Polynomial', all with degree = 1
    """

    def __init__(self, n_lines: int, n_compositions: int, n_variables: int) -> None:
        self.n_lines = n_lines
        self.n_compositions = n_compositions
        self.n_variables = n_variables
        self.quadratic_polynomials = [
            [Polynomial(1) for _ in range(n_compositions)] for __ in range(n_lines)]
        self.linear_combinations = [Polynomial(
            n_variables) for _ in range(n_lines)]

    def get_number_of_variables(self) -> int:
        return self.n_variables

    def get_number_of_lines(self) -> int:
        return self.n_lines

    def get_number_of_polynomials_per_line(self) -> int:
        return self.n_compositions

    def set_quadratic_polynomial(self, line_index: int, quadratic_polynomial_index: int, quadratic_polynomial: Polynomial) -> None:
        assert quadratic_polynomial.get_degree() == 2
        assert 0 <= line_index < self.n_lines
        assert 0 <= quadratic_polynomial_index < self.n_compositions
        self.quadratic_polynomials[line_index][quadratic_polynomial_index] = quadratic_polynomial

    def get_quadratic_polynomial(self, line_index: int, quadratic_polynomial_index: int) -> Polynomial:
        assert 0 <= line_index < self.n_lines
        assert 0 <= quadratic_polynomial_index < self.n_compositions
        return self.quadratic_polynomials[line_index][quadratic_polynomial_index]

    def set_linear_combination(self, line_index: int, linear_combination: Polynomial) -> None:
        assert linear_combination.get_degree() == 1
        assert 0 <= line_index < self.n_lines
        self.linear_combinations[line_index] = linear_combination

    def get_linear_combination(self, line_index: int) -> Polynomial:
        assert 0 <= line_index < self.n_lines
        return self.linear_combinations[line_index]

    def __str__(self) -> str:
        s = 'SECRET KEY:'
        s += '\n\n-> ' + str(self.n_lines) + ' lines (ie output variables)'
        s += '\n-> ' + str(self.n_compositions) + ' compositions per line'
        s += '\n-> ' + str(self.n_variables) + ' input variables'
        for line_index in range(self.n_lines):
            s += '\n\nline number '+str(line_index+1)+' :  '
            for quadratic_polynomial_index in range(self.n_compositions-1, -1, -1):
                quadratic_polynomial = self.quadratic_polynomials[line_index][quadratic_polynomial_index]
                quadratic_polynomial.set_alphabet(['X'])
                s += '(' + str(quadratic_polynomial) + ') o '
            linear_combination = self.linear_combinations[line_index]
            s += '(' + str(linear_combination) + ')'
        s += '\n\n'
        return s


class PublicKey:
    """A public key consists in a list of 'n_lines' (multivariate) 'Polynomial's.
    They all have 'n_variables' variables.
    """

    def __init__(self, n_lines: int, n_variables: int) -> None:
        self.n_lines = n_lines
        self.n_variables = n_variables
        self.polynomials = [Polynomial(n_variables) for _ in range(n_lines)]

    def get_number_of_variables(self) -> int:
        return self.n_variables

    def get_number_of_lines(self) -> int:
        return self.n_lines

    def set_polynomial(self, line_index: int, multivariate_polynomial: Polynomial) -> None:
        assert 0 <= line_index < self.n_lines
        self.polynomials[line_index] = multivariate_polynomial

    def get_polynomial(self, line_index: int) -> Polynomial:
        assert 0 <= line_index < self.n_lines
        return self.polynomials[line_index]

    def __str__(self) -> str:

        s = 'PUBLIC KEY:'
        s += '\n\n-> ' + str(self.n_lines) + ' lines (ie output variables)'
        s += '\n-> ' + str(self.n_variables) + ' input variables'
        for line_index in range(self.n_lines):
            polynomial = self.polynomials[line_index]
            s += '\n\nline number '+str(line_index+1)+' :  ' + str(polynomial)
        s += '\n\n'
        return s


def generate_random_quadratic_polynomial() -> Polynomial:
    pol = Polynomial(1)
    for coef_index in range(3):
        pol.set_coef((coef_index,), random.randint(1, specs.p-1))
    return pol


def generate_random_linear_combination(n_variables: int) -> Polynomial:
    lin = Polynomial(n_variables)
    lin.set_coef(get_tuple_index(-1, n_variables),
                 random.randint(1, specs.p-1))
    for variable_index in range(n_variables):
        lin.set_coef(get_tuple_index(variable_index, n_variables),
                     random.randint(1, specs.p-1))
    return lin


def generate_random_secret_key(n_lines: int, n_compositions: int, n_variables: int, verbose=False) -> SecretKey:
    if verbose:
        print('generating random secret key ...')
    secret_key = SecretKey(n_lines, n_compositions, n_variables)
    for line_index in range(n_lines):
        random_linear_combination = generate_random_linear_combination(
            n_variables)
        secret_key.set_linear_combination(
            line_index, random_linear_combination)
        for polynomial_index in range(n_compositions):
            random_quadratic_polynomial = generate_random_quadratic_polynomial()
            secret_key.set_quadratic_polynomial(
                line_index, polynomial_index, random_quadratic_polynomial)
    return secret_key


def generate_public_key(secret_key: SecretKey, verbose=False) -> PublicKey:
    if verbose:
        print('generating public key ...')
    public_key = PublicKey(secret_key.n_lines, secret_key.n_variables)
    for line_index in range(secret_key.n_lines):
        multivariate_polynomial = secret_key.get_linear_combination(line_index)
        for quadratic_polynomial_index in range(secret_key.n_compositions):
            quadratic_polynomial = secret_key.get_quadratic_polynomial(
                line_index, quadratic_polynomial_index)
            multivariate_polynomial = compose(
                quadratic_polynomial, multivariate_polynomial)
        public_key.set_polynomial(line_index, multivariate_polynomial)
    return public_key


def encrypt(message: Tuple, public_key: PublicKey) -> Tuple:
    """message : a tuple of integers"""
    assert len(message) == public_key.get_number_of_variables()
    encrypted_message = [0 for _ in range(public_key.get_number_of_lines())]
    for line_index in range(public_key.get_number_of_lines()):
        encrypted_message[line_index] = public_key.get_polynomial(
            line_index).eval(message)
    return tuple(encrypted_message)


def next_combination(combination: list, max_combination: List) -> bool:
    """Combination means here a list of integers.

    Returns false when the combination is the higher we can get,
    and cannot consequentally be increased.

    For instance:
    max_combination = (5, 7, 3, 4, 5, 5, 9)

    combination = (4, 6, 2, 1, 0, 1, 4) will become
    (0, 0, 0, 2, 0, 1, 4)

   """

    assert len(combination) == len(max_combination)
    for i in range(len(combination)):
        if combination[i] == max_combination[i] - 1:
            combination[i] = 0
        else:
            combination[i] += 1
            return True
    return False


def decrypt(encrypted_message: Tuple, secret_key: SecretKey):
    """Returns a set containing all the possible decrypted messages
    (tuple of int) correspnding to encrypted_message (tuple of int).
    """
    assert len(encrypted_message) == secret_key.get_number_of_lines()

    n_compositions = secret_key.get_number_of_polynomials_per_line()
    n_lines = secret_key.get_number_of_lines()
    n_variables = secret_key.get_number_of_variables()

    potential_values_for_linear_combinations = [[] for _ in range(n_lines)]

    for line_index in range(n_lines):
        values = [encrypted_message[line_index]]
        for pol_index in range(n_compositions - 1, -1, -1):
            quadratic_polynomial = secret_key.get_quadratic_polynomial(
                line_index, pol_index)
            new_values = []
            for v in values:
                new_values += solve_quadratic_polynomial(
                    quadratic_polynomial,
                    v)
            values = new_values
        potential_values_for_linear_combinations[line_index] = values

    solutions = []

    combination = [-1] + [0 for _ in range(-1 + n_lines)]
    max_combination = [len(potential_values_for_linear_combinations[line_index])
                       for line_index in range(n_lines)]
    linear_system = [secret_key.get_linear_combination(
        line_index) for line_index in range(n_lines)]

    while next_combination(combination, max_combination):
        values_eq = []
        for line_index in range(n_lines):
            val = potential_values_for_linear_combinations[line_index][combination[line_index]]
            values_eq.append(val)
        solutions += solve_linear_system_eq(linear_system, values_eq)
    return set([tuple(sol) for sol in solutions])
