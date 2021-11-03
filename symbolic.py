from typing import List, Tuple, Dict
from copy import deepcopy
from sympy.parsing import sympy_parser

from math_modulo import get_tuple_index
from protocol import generate_random_secret_key, generate_public_key

COEFS_ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
            'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


class SymbolicPolynomial:
    """
    This class represent a multivariate polynnomial (a polynomial
    with more than one variable), whose variables are strings on the form: 
        "(((a+b)*d) - k)"

    For the polynomial of 3 variables :
    (a-b) + b.X + (c*c).X.Z + d.Y² + (e+(e+r)).X.Z² + e.Y³

    self.n_variables = 3

    self.coefs[(0, 0, 0)] = "(a-b)"
    self.coefs[(1, 0, 0)] = "b"
    self.coefs[(1, 0, 1)] = "(c*c)"
    self.coefs[(0, 2, 0)] = "d"
    self.coefs[(1, 0, 2)] = "(e+(e+r))"
    self.coefs[(0, 0, 3)] = "e"
    """

    def __init__(self, n_variables: int) -> None:
        self.coefs: Dict[Tuple, str] = {}
        self.n_variables = n_variables

    def get_number_of_variables(self) -> int:
        return self.n_variables

    def get_coefs(self) -> Dict[Tuple, str]:
        return self.coefs

    def set_coef(self, coef_index: Tuple, coef_value: str) -> None:
        assert len(coef_index) == self.n_variables
        self.coefs[coef_index] = coef_value

    def get_coef(self, coef_index: Tuple) -> str:
        assert len(coef_index) == self.n_variables
        if coef_index in self.coefs:
            return self.coefs[coef_index]
        else:
            return "0"

    def get_degree(self) -> int:
        # a bit slow, do not use it too much
        degree = -1
        for index in self.coefs:
            degree = max(degree, sum(index))
        return degree

    def get_coefs_on_list(self):
        """Used when the Polynomial has degree 1, that is to say :
        (b-a).x + (b*t).y + c.z + d
        returns : 
        ["(b-a)", "(b*t)", "c", "d"]
        """
        assert self.get_degree() <= 1

        coefs_on_list = []
        for coef_number in range(self.n_variables):
            coefs_on_list.append(self.get_coef(
                get_tuple_index(coef_number, self.n_variables)))
        coefs_on_list.append(self.get_coef(
            get_tuple_index(-1, self.n_variables)))  # constant coef
        return coefs_on_list

    def __str__(self) -> str:
        VARIABLES_ALPHABET = ['X', 'Y', 'Z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                    'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W']
        assert self.n_variables <= len(VARIABLES_ALPHABET)
        polynomial_str = ""
        for coef_index, coef_value in self.coefs.items():
            coef_str = str(coef_value)
            if sum(coef_index) != 0:
                coef_str += '.'
            for var_index in range(self.n_variables):
                if coef_index[var_index] != 0:
                    coef_str += VARIABLES_ALPHABET[var_index]
                    if coef_index[var_index] != 1:
                        coef_str += "^" + str(coef_index[var_index])
            polynomial_str += coef_str + " + "
        return polynomial_str[0:-2]


def add_symbolic_polynomials(a: SymbolicPolynomial, b: SymbolicPolynomial) -> SymbolicPolynomial:
    """Returns the sum of 2 symbolic multivariate polynomials
    They must have the same number of unknows"""
    assert a.get_number_of_variables() == b.get_number_of_variables()
    c = SymbolicPolynomial(a.get_number_of_variables())
    for coef_index in set(a.get_coefs().keys()).union(set(b.get_coefs().keys())):
        c.set_coef(coef_index, "(" + a.get_coef(coef_index) +
                   " + " + b.get_coef(coef_index) + ")")
    return c


def multiply_symbolic_polynomials(a: SymbolicPolynomial, b: SymbolicPolynomial) -> SymbolicPolynomial:
    """Returns the product of 2 symbolic multivariate polynomials
    They must have the same number of unknows"""
    assert a.get_number_of_variables() == b.get_number_of_variables()
    c = SymbolicPolynomial(a.get_number_of_variables())
    for a_coef_index, a_coef_value in a.get_coefs().items():
        for b_coef_index, b_coef_value in b.get_coefs().items():
            # element wise tuple addition
            new_coef_index = tuple(map(sum, zip(a_coef_index, b_coef_index)))
            new_coef_value = "(" + a_coef_value + "*" + b_coef_value + ")"
            new_coef_value = "(" + new_coef_value + "+" + \
                c.get_coef(new_coef_index) + ")"
            c.set_coef(new_coef_index, new_coef_value)
    return c


def power_symbolic_multivariate_polynom(a: SymbolicPolynomial, n: int) -> SymbolicPolynomial:
    # Returns a raised to the power n (a*...*a n times)
    assert n >= 0
    c = SymbolicPolynomial(a.get_number_of_variables())
    c.set_coef(get_tuple_index(-1, a.get_number_of_variables()), "1")
    for _ in range(n):
        c = multiply_symbolic_polynomials(c, a)
    return c


def multiply_scalar_symbolic_polynom(alpha: str, a: SymbolicPolynomial) -> SymbolicPolynomial:
    # Returns the product of the symbolic multivariate polynomial a by the scalar alpha (modulo 'mod')
    c = SymbolicPolynomial(a.get_number_of_variables())
    for coef_index, coef_value in a.get_coefs().items():
        new_value = "(" + coef_value + " * " + alpha + ")"
        c.set_coef(coef_index, new_value)
    return c


def compose_symbolic(a: SymbolicPolynomial, b: SymbolicPolynomial) -> SymbolicPolynomial:
    """returns the composition a∘b = a(b)
        a must contain only one variable"""

    assert a.get_number_of_variables() == 1
    c = SymbolicPolynomial(b.get_number_of_variables())
    c.set_coef(get_tuple_index(-1, b.get_number_of_variables()),
               a.get_coef((0,)))
    for power in range(1, a.get_degree() + 1):
        d = power_symbolic_multivariate_polynom(b, power)
        d = multiply_scalar_symbolic_polynom(a.get_coef((power,)), d)
        c = add_symbolic_polynomials(c, d)
    return c


alphabet_index = 0
def get_letter() -> str:
    global alphabet_index
    alphabet_index += 1
    return COEFS_ALPHABET[alphabet_index-1]


def generate_symbolic_linear_combination(n_variables: int) -> SymbolicPolynomial:
    symbolic_linear_combination = SymbolicPolynomial(n_variables)
    for variable_index in range(n_variables):
        symbolic_linear_combination.set_coef(
            get_tuple_index(variable_index, n_variables), get_letter())

    symbolic_linear_combination.set_coef(
        get_tuple_index(-1, n_variables), get_letter())
    return symbolic_linear_combination


def generate_symbolic_quadratic_polynomial() -> SymbolicPolynomial:
    symbolic_quadratic_polynomial = SymbolicPolynomial(1)
    for i in range(3):
        symbolic_quadratic_polynomial.set_coef((i,), get_letter())
    return symbolic_quadratic_polynomial


def generate_symbolic_public_key_polynomial(n_variables, n_compositions) -> SymbolicPolynomial:
    symbolic_polynomial = generate_symbolic_linear_combination(n_variables)

    for _ in range(n_compositions):
        symbolic_quadratic_polynomial = generate_symbolic_quadratic_polynomial()
        symbolic_polynomial = compose_symbolic(
            symbolic_quadratic_polynomial, symbolic_polynomial)
    return symbolic_polynomial


def generate_system_needed_to_break_secret_key_security(n_compositions: int, n_variables: int):
    """Recovering the public key from the secret key is equivalent to 
    solving a system of multivariate polynomials. The polynomials are
    returned as a list of sympy polynomials."""

    secret_key = generate_random_secret_key(1, n_compositions, n_variables)
    public_key = generate_public_key(secret_key)

    polynomial_with_integers_coefs = public_key.polynomials[0]
    polynomial_with_symbolic_coefs = generate_symbolic_public_key_polynomial(
        n_variables, n_compositions)

    system_needed_to_break_secret_key_security = []
    for coef_index in polynomial_with_symbolic_coefs.coefs:
        expr = polynomial_with_symbolic_coefs.get_coef(coef_index)
        expr += " - " + \
            str(polynomial_with_integers_coefs.get_coef(coef_index))
        sympy_expr = sympy_parser.parse_expr(expr)
        system_needed_to_break_secret_key_security.append(sympy_expr)

    return system_needed_to_break_secret_key_security


def pretty_print_polyomial_system(polyomial_system: List) -> None:
    for polynomial in polyomial_system:
        pretty_polynomial_str = str(polynomial)
        pretty_polynomial_str = pretty_polynomial_str.replace('**', '^')
        pretty_polynomial_str = pretty_polynomial_str.replace('*', '.')
        print(pretty_polynomial_str)
