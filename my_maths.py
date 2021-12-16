import random
from typing import Any, Tuple, Dict, List, Union
from copy import deepcopy
from itertools import combinations

import specs

"""
All the math needed to work with polynomials
(of potentially multiple variables) modulo p.
"""


class Polynomial:
    """
    This class represents a multivariate polynomial (a polynomial
    with one ore more variable) modulo p :
    3 + 12.X + 32X.Y³ + 41.Y.Z³ + ...
    This class can also represent a univariate polynomial
    (when n_variables == 1):
    2 + X + 4.X² + 18.X³ + ...
    In that case, we can compose it with another multivariate
    polynomial (see the 'compose' function).
    It can also represent a linear combination (when degree <= 1):
    4 + 98.X + 67.Y + 2.Z + ...

    The coefficients are stored in the dict self.coefs
    Most of the time, they are integers (modulo p), but they can
    have other type: we can consider a 'Polynomial' where its coefficients
    are 'Polynomial' too.

    For instance:

    For the polynomial of 3 variables :
    7 + 2.X + 13.X.Z + Y² + 16.X.Z² + 19.Z³

    self.n_variables = 3

    self.coefs[(0, 0, 0)] = 7
    self.coefs[(1, 0, 0)] = 2
    self.coefs[(1, 0, 1)] = 13
    self.coefs[(0, 2, 0)] = 1
    self.coefs[(1, 0, 2)] = 16
    self.coefs[(0, 0, 3)] = 19
    """

    def __init__(self, n_variables: int) -> None:
        self.n_variables = n_variables
        self.coefs: Dict[Tuple, Any] = {}

    def get_number_of_variables(self) -> int:
        return self.n_variables

    def get_coefs(self) -> Dict[Tuple, Any]:
        return self.coefs

    def set_coef(self, coef_index: Tuple, coef_value: Any) -> None:
        assert len(coef_index) == self.n_variables
        self.coefs[coef_index] = coef_value

    def get_coef(self, coef_index: Tuple) -> Any:
        assert len(coef_index) == self.n_variables
        if coef_index in self.coefs:
            return self.coefs[coef_index]
        else:
            return 0

    def get_degree(self) -> int:
        degree = -1
        for index in self.coefs:
            degree = max(degree, sum(index))
        return degree

    def drop_coef(self, coef_index: Tuple) -> int:
        assert coef_index in self.get_coef
        del self.coefs[coef_index]

    def clean(self):
        """Remove the unused final variables.

        Example:

        self.coefs = {
            (1, 4, 0, 2, 0, 0, 0): 16 ,
            (5, 0, 0, 1, 1, 0, 0): 3 ,
            (5, 3, 0, 4, 0, 0, 0): 11
        }

        will become:

         self.coefs = {
            (1, 4, 0, 2, 0,): 16 ,
            (5, 0, 0, 1, 1): 3 ,
            (5, 3, 0, 4, 0): 11
        }
        """
        last_variable_used_biggest_index = -1
        for coef_index, coef_value in self.coefs.items():
            if coef_value == 0:
                pass
            last_variable_used_index = -1
            for var_index in range(self.n_variables - 1, -1, -1):
                if coef_index[var_index] != 0:
                    last_variable_used_index = var_index
                    break
            if last_variable_used_index > last_variable_used_biggest_index:
                last_variable_used_biggest_index = last_variable_used_index

        coef_indexes = list(self.coefs)
        for coef_index in coef_indexes:
            coef_value = self.coefs[coef_index]
            del self.coefs[coef_index]
            if coef_value != 0:
                new_coef_index = tuple(
                    list(coef_index)[0: last_variable_used_biggest_index + 1]
                )
                self.coefs[new_coef_index] = coef_value
        self.n_variables = last_variable_used_biggest_index + 1

    def remove_zero_coefs(self):
        """34.X.Y² + 0.Y.Z + 56.Z³ + ...
        will become -> 34.X.Y² + 56.Z³ + ..."""
        indexes_with_zero_values = []
        for index, value in self.coefs.items():
            if value == 0:
                indexes_with_zero_values.append(index)
        for index in indexes_with_zero_values:
            del self.coefs[index]

    def __str__(self, alphabet: List[str] = specs.alphabet1_extended) -> str:
        assert self.n_variables <= len(alphabet)
        if self.n_variables == 1:
            alphabet = ["X"]
        polynomial_str = ""
        for coef_index, coef_value in self.coefs.items():
            if type(coef_value) == Polynomial:
                coef_str = (
                    "("
                    + coef_value.__str__(alphabet=specs.alphabet2_extended)
                    + ")"
                )
            else:
                coef_str = str(coef_value)
            if sum(coef_index) != 0:
                coef_str += "."
            for var_index in range(self.n_variables):
                if coef_index[var_index] != 0:
                    coef_str += alphabet[var_index]
                    if coef_index[var_index] != 1:
                        coef_str += "^" + str(coef_index[var_index])
            polynomial_str += coef_str + " + "
        polynomial_str = polynomial_str[0:-3]
        return polynomial_str

    def __repr__(self):
        return str(self)


"""In this protocol, every mathematic object is either an integer (modulo p),
either a Polynomial"""
Value = Union[int, Polynomial]


def get_neutral_value(value_sample: Value) -> Value:
    """Returns the equivalent of 1, with good type."""
    assert isinstance(value_sample, Value.__args__)
    if type(value_sample) == int:
        return 1
    if type(value_sample) == Polynomial:
        polynomial = Polynomial(value_sample.get_number_of_variables())
        neutral_coef = 1
        polynomial.set_coef(
            get_tuple_index(-1, value_sample.get_number_of_variables()),
            neutral_coef,
        )
        return polynomial
    assert False


def get_null_value(value_sample: Value) -> Value:
    """Returns the equivalent of 0, with good type."""
    assert isinstance(value_sample, Value.__args__)
    if type(value_sample) == int:
        return 0
    if type(value_sample) == Polynomial:
        polynomial = Polynomial(value_sample.get_number_of_variables())
        return polynomial
    assert False


def get_random_coef():
    return random.randint(0, specs.p - 1)


class Vector:
    """A fixed number of values (int (modulo p) or polynomial)."""

    def __init__(self, length: int = None, coefs: List[Value] = None) -> None:
        assert length is not None or coefs is not None
        if coefs is not None:
            self.coefs = coefs
            self.length = len(coefs)
        else:
            self.length = length
            self.coefs: List[Value] = [None] * length

    def __getitem__(self, index: int) -> Value:
        assert type(index) == int
        assert 0 <= index <= self.length - 1
        return self.coefs[index]

    def __setitem__(self, index: int, value: Value):
        assert type(index) == int
        assert 0 <= index <= self.length - 1
        self.coefs[index] = value

    def __len__(self):
        return len(self.coefs)

    def __str__(self) -> str:
        s = "["
        for index in self.coefs:
            s += str(index) + ", "
        s = s[:-2] + "]"
        return s

    def __repr__(self):
        return str(self)

    def __eq__(self, obj):
        return isinstance(obj, Vector) and obj.coefs == self.coefs


def insert_in_vector(
    vector: Vector, insertion_index: int, value: int
) -> Vector:
    """
    Inserts a coef at a given position and returns a new vector.
    ex:
    vector.coefs = [4, 1, 31, 8, 2, 3]
    index = 3
    value = 50

    returns -> a new vector with coefs = [4, 1, 31, 50, 8, 2, 3]
    """
    assert 0 <= insertion_index <= len(vector)

    result = Vector(length=len(vector) + 1)
    for before_index in range(insertion_index):
        result[before_index] = vector[before_index]
    result[insertion_index] = value
    for after_index in range(insertion_index, len(vector)):
        result[after_index + 1] = vector[after_index]
    return result


def add(a: Value, b: Value) -> Value:
    result = None
    if a == 0:
        return b
    elif b == 0:
        return a
    elif type(a) == int and type(b) == int:
        """add two integers modulo p"""
        return (a + b) % specs.p
    elif type(a) == int and type(b) == Polynomial:
        c = deepcopy(b)
        constant_index = get_tuple_index(-1, c.get_number_of_variables())
        increase_coef_by(c, constant_index, a)
        result = c
    elif type(a) == Polynomial and type(b) == int:
        result = add(b, a)
    elif type(a) == Polynomial and type(b) == Polynomial:
        """Returns the sum of 2 multivariate polynomials
        They must have the same number of unknowns"""
        assert a.get_number_of_variables() == b.get_number_of_variables()
        c = Polynomial(a.get_number_of_variables())
        for coef_index in set(a.get_coefs().keys()).union(
            set(b.get_coefs().keys())
        ):
            c.set_coef(
                coef_index, add(a.get_coef(coef_index), b.get_coef(coef_index))
            )
        result = c
    assert result is not None
    return result


def substract(a: Value, b: Value) -> Value:
    result = None
    if type(a) == int and type(b) == int:
        """add two integers modulo p"""
        result = (a - b) % specs.p
    elif type(a) == Polynomial and type(b) == int:
        c = deepcopy(a)
        constant_index = get_tuple_index(-1, c.get_number_of_variables())
        decrease_coef_by(c, constant_index, b)
        result = c
    elif type(a) == int and type(b) == Polynomial:
        result = add(a, multiply(-1, b))
    elif type(a) == Polynomial and type(b) == Polynomial:
        """Returns the difference of 2 multivariate polynomials
        They must have the same number of unknowns"""
        assert a.get_number_of_variables() == b.get_number_of_variables()
        c = Polynomial(a.get_number_of_variables())
        for coef_index in set(a.get_coefs().keys()).union(
            set(b.get_coefs().keys())
        ):
            c.set_coef(
                coef_index,
                substract(a.get_coef(coef_index), b.get_coef(coef_index)),
            )
        result = c

    assert result is not None
    return result


def multiply(a: Value, b: Value, a_is_scalar=False) -> Value:
    result = None
    if type(a) == int and type(b) == int:
        """multiply two integers modulo p"""
        result = (a * b) % specs.p
    elif type(a) == int and type(b) == Polynomial:
        """Returns the product of the multivariate polynomial a by the scalar
        alpha"""
        c = Polynomial(b.get_number_of_variables())
        for coef_index, coef_value in b.get_coefs().items():
            new_value = multiply(a, coef_value)
            c.set_coef(coef_index, new_value)
        result = c
    elif type(a) == Polynomial and type(b) == int:
        result = multiply(b, a)
    elif type(a) == Polynomial and type(b) == Polynomial:
        if a_is_scalar:
            c = Polynomial(b.get_number_of_variables())
            for coef_index, coef_value in b.get_coefs().items():
                new_coef_value = multiply(a, coef_value)
                c.set_coef(coef_index, new_coef_value)
            result = c
        else:
            """Returns the product of 2 multivariate polynomials
            They must have the same number of unknowns"""

            assert a.get_number_of_variables() == b.get_number_of_variables()
            c = Polynomial(a.get_number_of_variables())
            for a_coef_index, a_coef_value in a.get_coefs().items():
                for b_coef_index, b_coef_value in b.get_coefs().items():
                    # element wise tuple addition
                    new_coef_index = tuple(
                        map(sum, zip(a_coef_index, b_coef_index))
                    )
                    new_coef_value = multiply(a_coef_value, b_coef_value)
                    new_coef_value = add(
                        new_coef_value, c.get_coef(new_coef_index)
                    )
                    c.set_coef(new_coef_index, new_coef_value)
            result = c

    assert result is not None
    return result


def power(a: Value, b: int) -> Value:
    assert type(b) == int
    if type(a) == int:
        """exponentiation of 2 integers modulo p"""
        return pow(a, b, specs.p)
    if type(a) == str:
        return "(" + a + ")^" + b
    if type(a) == Polynomial:
        """Returns the polynomial in parameter, raised to the power n"""
        assert b >= 0
        c = Polynomial(a.get_number_of_variables())
        c.set_coef(tuple([0 for _ in range(a.get_number_of_variables())]), 1)
        for _ in range(b):
            c = multiply(c, a)
        return c


def inverse(a: int) -> int:
    assert type(a) == int
    """returns b such that a*b = 1 mod p (a != 0 mod p)"""
    return pow(a, -1, specs.p)


def divide(a: int, b: int) -> int:
    assert type(a) == int and type(b) == int
    """division of two integers mod p (b != 0 mod p)"""
    return multiply(a, inverse(b))


def compose(
    univariate_polynomial: Polynomial, multivariate_polynomial: Polynomial
) -> Polynomial:
    """returns the composition a∘b = a(b)
    a must contain only one variable"""
    assert univariate_polynomial.get_number_of_variables() == 1
    n_variables = multivariate_polynomial.get_number_of_variables()
    new_multivariate_polynomial = Polynomial(n_variables)
    new_multivariate_polynomial.set_coef(
        tuple([0 for _ in range(n_variables)]),
        univariate_polynomial.get_coef((0,)),
    )
    for _power in range(1, univariate_polynomial.get_degree() + 1):
        powered_polynomial = power(multivariate_polynomial, _power)
        powered_polynomial = multiply(
            univariate_polynomial.get_coef((_power,)),
            powered_polynomial,
            a_is_scalar=True,
        )
        new_multivariate_polynomial = add(
            new_multivariate_polynomial, powered_polynomial
        )
    return new_multivariate_polynomial


def get_tuple_index(int_index: int, n_variables: int):
    """An index of coefficient of a 'Polynomial' (see
    below) is represented by a tuple.
    For instance :
    polynomial = ... + 187.X³.Y⁰.Z⁴ + ... will result in
    polynomial.coefs[ (3, 0, 4) ] = 187

    In the protocol, we need to use linear combinations
    of variables: 16 + 3.X + 18.Y + 12.Z + ...

    In that case a coefficient index has the form :
    (0, 0, ... , 0, 1, 0, ... , 0)

    The goal of this function is to easily generate such indexes.
    This function returns a tuples of length (param) 'n_var', full
    of zero except in position (param) 'int_index', where it's a 1.
    If 'int_index' == -1, (0, 0, 0, ... , 0) is returned.

    int_index == -1 -> returns (0, 0, 0, 0, ... , 0)
    int_index == 0  -> returns (1, 0, 0, 0, ... , 0)
    int_index == 1  -> returns (0, 1, 0, 0, ... , 0)
    int_index == 2  -> returns (0, 0, 1, 0, ... , 0)
    ..."""

    tpl = [0] * n_variables
    if int_index == -1:
        return tuple(tpl)
    else:
        tpl[int_index] = 1
        return tuple(tpl)


def increase_coef_by(
    polynomial: Polynomial, coef_index: Tuple, coef_value: Value
) -> None:
    """Increase the coefficient of a polynomial by a given value.
    This modifies the Polynomial in parameter"""
    assert len(coef_index) == polynomial.n_variables
    polynomial.coefs[coef_index] = add(
        polynomial.get_coef(coef_index), coef_value
    )


def decrease_coef_by(
    polynomial: Polynomial, coef_index: Tuple, coef_value: Value
) -> None:
    """Decrease the coefficient of a polynomial by a given value.
    This modifies the Polynomial in parameter"""
    assert len(coef_index) == polynomial.n_variables
    polynomial.coefs[coef_index] = substract(
        polynomial.get_coef(coef_index), coef_value
    )


def get_coef_indexes(degree: int, n_variables: int) -> List[Tuple]:
    """Returns the list of all the coefficient indexes,
    with exactly the degree given in parameter

    Example: degree = 2, n_variables = 3
    Returns : [(2, 0, 0), (1 ,1, 0), (1, 0, 1),
                (0, 2, 0), (0, 1, 1), (0, 0, 2)]
    """
    coef_indexes = []
    t = degree + n_variables - 1
    for comb in combinations(range(t), n_variables - 1):
        index = [b - a - 1 for a, b in zip((-1,) + comb, comb + (t,))]
        coef_indexes.append(tuple(index))
    return coef_indexes


def eval_polynomial(polynomial: Polynomial, input: Vector) -> int:
    """Evaluate the polynomial with the distribution of values given
    in parameter 'variables'

    Implements the method described in the article:
    - Evaluate first all the monomials without considering the coefficients of
    the polynomials (compute X³Y²Z⁵ , not 134.X³Y²Z⁵).
    - To do so, start from the monomials of low degree.
    - Compute the monomials (without coefficient) of degree k + 1 from those
    of degree k (stored in memory), as it only involves one multiplication: to
    compute X³Y²Z⁵ , multiply X by X²Y²Z⁵ (already calculated)."""
    assert len(input) == polynomial.get_number_of_variables()
    monomial_values = {}  # without the coefficients from the polynomial
    monomial_values[
        get_tuple_index(-1, polynomial.get_number_of_variables())
    ] = get_neutral_value(input[0])

    for degree in range(1, 1 + polynomial.get_degree()):
        coef_indexes = get_coef_indexes(
            degree, polynomial.get_number_of_variables()
        )
        for index in coef_indexes:
            for variable_index in range(polynomial.get_number_of_variables()):
                if index[variable_index] != 0:
                    previous_index = list(index)
                    previous_index[variable_index] -= 1
                    previous_index = tuple(previous_index)
                    previous_value = monomial_values[previous_index]
                    value = multiply(previous_value, input[variable_index])
                    monomial_values[index] = value

                    break
    result = get_null_value(input[0])

    for coef_index, coef_value in polynomial.coefs.items():
        term = multiply(
            coef_value, monomial_values[coef_index], a_is_scalar=True
        )
        result = add(result, term)

    return result


def eval_one_variable(polynomial: Polynomial, var_index: int, value: int):
    """returns the polynomial we get by fixing the value of one variable"""
    assert 0 <= var_index < polynomial.get_number_of_variables()
    new_polynomial = Polynomial(polynomial.get_number_of_variables() - 1)
    for coef_index, coef_value in polynomial.coefs.items():
        new_coef_index = list(coef_index)
        del new_coef_index[var_index]
        new_coef_index = tuple(new_coef_index)
        new_coef_value = multiply(
            coef_value, power(value, coef_index[var_index]), a_is_scalar=True
        )
        increase_coef_by(new_polynomial, new_coef_index, new_coef_value)
    return new_polynomial


def square_root(a: int):
    assert type(a) == int
    assert 0 <= a <= specs.p - 1
    """Implementation of the Tonelli-Shanks algorithm.

    code inspired by :
    https://gist.github.com/nakov/60d62bdf4067ea72b7832ce9f71ae079

    Fnds x such as x^2 = a (modulo p)
    Returns [x, p-x] the two solutions
    [None, None] is returned is no square root exists.
    """

    def legendre_symbol(a):
        ls = power(a, (specs.p - 1) // 2)
        return -1 if ls == specs.p - 1 else ls

    def enhance_return(one_square_root):
        if one_square_root is None:
            return [None, None]

        one_square_root = one_square_root % specs.p
        if one_square_root == 0:
            return [0, 0]
        assert 0 <= one_square_root < specs.p
        return [one_square_root, specs.p - one_square_root]

    if a == 0:
        return enhance_return(0)
    elif legendre_symbol(a) != 1:
        return enhance_return(None)
    elif specs.p == 2:
        return [a, a]
    elif specs.p % 4 == 3:
        return enhance_return(power(a, (specs.p + 1) // 4))
    s = specs.p - 1
    e = 0
    while s % 2 == 0:
        s //= 2
        e += 1
    n = 2
    while legendre_symbol(n) != -1:
        n += 1
    x = power(a, (s + 1) // 2)
    b = power(a, s)
    g = power(n, s)
    r = e

    while True:
        t = b
        m = 0
        for m in range(r):
            if t == 1:
                break
            t = power(t, 2)
        if m == 0:
            return enhance_return(x)
        gs = power(g, 2 ** (r - m - 1))
        g = power(gs, 2)
        x = multiply(x, gs)
        b = multiply(b, g)
        r = m


def solve_linear_system(linear_system: List[Polynomial]) -> List[Vector]:
    """Linear system solver with Gaussian elimination:

    If we want to solve the following system :

    (mod 23)
    2x + 3y = 13
    4x + 9y = 12
    3x + 5y = 21

    The parameter 'linear_system' must be :
    [Polynomial(2x + 3y - 13), Polynomial(4x + 9y - 44),
    Polynomial(3x + 5y - 26)]

    -> returns the list of all the solutions, each solution beeing
    represented by a Vector

    In this example : [Vector([2, 3])] is returned because x=2, y=3 is the only
    solution.

    (Returns [] if there is no solutions)
    """
    for polynomial in linear_system:
        assert polynomial.get_degree() <= 1

    n_variables = linear_system[0].get_number_of_variables()
    n_lines = len(linear_system)

    for var_index in range(0, n_variables):

        # looking for a non zero pivot
        line_index = var_index
        pivot = 0
        while line_index < n_lines and pivot == 0:
            pivot = linear_system[line_index].get_coef(
                get_tuple_index(var_index, n_variables)
            )
            if pivot != 0:
                # invert lines
                temp = linear_system[line_index]
                linear_system[line_index] = linear_system[var_index]
                linear_system[var_index] = temp
            line_index += 1
        if pivot != 0:
            inv = inverse(pivot)
            linear_system[var_index] = multiply(inv, linear_system[var_index])

            for line_index_2 in range(var_index + 1, n_lines):
                alpha = multiply(
                    -1,
                    linear_system[line_index_2].get_coef(
                        get_tuple_index(var_index, n_variables)
                    ),
                )
                temp = multiply(alpha, linear_system[var_index])
                linear_system[line_index_2] = add(
                    linear_system[line_index_2], temp
                )

        else:

            solutions = []
            for val in range(specs.p):
                # the variable var_index receives the value 'val'
                new_linear_system = [
                    eval_one_variable(pol, var_index, val)
                    for pol in linear_system
                ]
                temp_solutions = solve_linear_system(new_linear_system)
                if temp_solutions == []:
                    break
                for temp_sol in temp_solutions:
                    solutions.append(
                        insert_in_vector(temp_sol, var_index, val)
                    )
            return solutions

    for h in range(n_variables, n_lines):
        if linear_system[h].get_coef(get_tuple_index(-1, n_variables)) != 0:
            return []  # no solutions

    sol = Vector(length=n_variables)
    for line_index in range(n_variables - 1, -1, -1):
        sum = linear_system[line_index].get_coef(
            get_tuple_index(-1, n_variables)
        )
        for var_index_2 in range(line_index + 1, n_variables):
            sum += multiply(
                sol[var_index_2],
                linear_system[line_index].get_coef(
                    get_tuple_index(var_index_2, n_variables)
                ),
            )
        sol[line_index] = multiply(-1, sum)
    return [sol]


def solve_quadratic_polynomial(quadratic_polynomial: Polynomial, d: int):
    """quadratic_polynomial : Polynomial of the form  ax² + bx + c
    Returns the list of the solutions of ax² + bx + c = d (modulo p).
    """

    assert quadratic_polynomial.get_number_of_variables() == 1
    assert quadratic_polynomial.get_degree() <= 2

    a = quadratic_polynomial.get_coef((2,))
    b = quadratic_polynomial.get_coef((1,))
    c = quadratic_polynomial.get_coef((0,))

    if a == 0 and b == 0:
        if c == d:
            return [x for x in range(specs.p)]
        else:
            return []

    if specs.p == 2:
        # This case should not be used in this protocol.
        # Indeed, as x^2 = x (mod 2), the security of the
        # quadratic compositions is no more guaranteed.
        if add(a, b) == 0:
            if c == d:
                return [0, 1]
            else:
                return []
        else:
            return divide(substract(d, c), add(a, b))

    if a == 0:
        return [multiply(inverse(b), substract(d, c))]

    else:
        b = multiply(inverse(a), b)
        c = multiply(inverse(a), c)
        d = multiply(inverse(a), d)
        e = substract(d, c)
        f = divide(b, 2)
        g = add(e, power(f, 2))
        h = square_root(g)
        if h[0] is None:
            return []
        else:
            return [substract(h[0], f), substract(h[1], f)]
