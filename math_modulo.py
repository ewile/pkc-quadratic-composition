from sympy import groebner
from sympy.polys.domains import FiniteField
from typing import Tuple, Dict, List
from copy import deepcopy
from itertools import combinations

import specs

"""
All the math needed to work with polynomials 
(of potentially multiple variables) in Fp, the finite 
field of p elements, that is to say working modulo p.
"""


def add(a: int, b: int) -> int:
    """add two integers modulo p"""
    return (a + b) % specs.p


def substract(a: int, b: int) -> int:
    """substract two integers modulo p"""
    return (a - b) % specs.p


def multiply(a: int, b: int) -> int:
    """multiply two integers modulo p"""
    return a * b % specs.p


def power(a: int, b: int) -> int:
    """exponentiation of 2 integers modulo p"""
    return pow(a, b, specs.p)


def inverse(a: int) -> int:
    """returns b such that a*b = 1 mod p (a != 0 mod p)"""
    return pow(a, -1, specs.p)


def divide(a: int, b: int) -> int:
    """division of two integers mod p (b != 0 mod p)"""
    return multiply(a, inverse(b))


def get_tuple_index(int_index: int, n_variables: int):
    """An index of of coefficent of a 'Polynomial' (see 
    below) is represented by a tuple. 
    For instance :
    polynomial = ... + 187.X³.Y⁰.Z⁴ + ... will result in
    polynomial.coefs[ (3, 0, 4) ] = 187

    In the protocol, we need to use linear combinations
    of variables: 16 + 3.X + 18.Y + 12.Z + ...

    In that case a coefficent index has the form :
    (0, 0, ... , 0, 1, 0, ... , 0)

    The goal of this function is to easily generate such indexes.
    This function returns a tuples of length (param) 'n_var', full
    of zero except in position (param) 'int_index', where it's a 1.
    If 'int_index' == -1, (0, 0, 0, ... , 0) is returned.

    int_index == -1 -> (0, 0, 0, 0, ... , 0)
    int_index == 0  -> (1, 0, 0, 0, ... , 0)
    int_index == 1  -> (0, 1, 0, 0, ... , 0)
    int_index == 2  -> (0, 0, 1, 0, ... , 0)
    ...    """

    tpl = [0]*n_variables
    if int_index == -1:
        return tuple(tpl)
    else:
        tpl[int_index] = 1
        return tuple(tpl)


def get_coef_indexes(degree: int, n_variables: int) -> List[Tuple]:
    """Returns the list of all the ceofficient indexes,
    with exactlt the degree given in parameter

    Example: degree = 2, n_variables = 3
    Returns : [(2, 0, 0), (1 ,1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)]
    """
    coef_indexes = []
    t = degree + n_variables - 1
    for comb in combinations(range(t), n_variables - 1):
        index = [b - a - 1 for a, b in zip((-1,) + comb, comb + (t,))]
        coef_indexes.append(tuple(index))
    return coef_indexes


class Polynomial:
    """
    This class represent a multivariate polynnomial (a polynomial
    with one ore more variable) modulo p :
    3 + 12.X + 32X.Y³ + 41.Y.Z³ + ...
    This class can also represent a univariate polynomial 
    (when n_variables == 1):
    2 + X + 4.X² + 18.X³ + ...
    In that case, we can compose it with another multivariate
    polynnomial (see the 'compose' function). 
    It can also represent a linear combination (when degree <= 1):
    4 + 98.X + 67.Y +2.Z + ...

    The coeficients are stored in the dict self.coefs
    They are integers, but must be seen modulo p

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

    # symbols used to display the variables of the polynomial
    default_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                        'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    def __init__(self, n_variables: int) -> None:
        self.coefs: Dict[Tuple, int] = {}
        self.n_variables = n_variables
        self.alphabet = Polynomial.default_alphabet

    def get_number_of_variables(self) -> int:
        return self.n_variables

    def get_coefs(self) -> Dict[Tuple, int]:
        return self.coefs

    def set_coef(self, coef_index: Tuple, coef_value: int) -> None:
        assert 0 <= coef_value < specs.p
        assert len(coef_index) == self.n_variables
        self.coefs[coef_index] = coef_value

    def increase_coef_by(self, coef_index: Tuple, coef_value: int) -> None:
        assert 0 <= coef_value < specs.p
        assert len(coef_index) == self.n_variables
        self.coefs[coef_index] = add(self.get_coef(coef_index), coef_value)

    def get_coef(self, coef_index: Tuple) -> int:
        assert len(coef_index) == self.n_variables
        if coef_index in self.coefs:
            return self.coefs[coef_index]
        else:
            return 0

    def eval_naive(self, variables: Tuple) -> int:
        """Naive implementation. For better performances,
        use 'eval' just below.

        Evaluate the polynomial with the distribution of values given
        in parameter 'variables' """
        assert len(variables) == self.n_variables
        result = 0
        for coef_index, coef_value in self.coefs.items():
            term = coef_value
            for variable_index, variable_power in enumerate(coef_index):
                term = multiply(term, power(
                    variables[variable_index], variable_power))
            result = add(result, term)
        return result

    def eval(self, variables: Tuple) -> int:
        """Evaluate the polynomial with the distribution of values given
        in parameter 'variables' 

        Implements the method described in the article:
        - Evaluate first all the monomials without considering the coefficients of the polynomials (compute
        X³Y²Z⁵ , not 134.X³Y²Z⁵).
        - To do so, start from the monomials of low degree.
        - Compute the monomials (without coefficient) of degree k + 1 from those of degree k (stored in
        memory), as it only involves one multiplication: to compute X³Y²Z⁵ , multiply X by X²Y²Z⁵
        (already calculated)."""
        assert len(variables) == self.n_variables

        monomial_values = {}  # without the coefficents from the polynomial
        monomial_values[get_tuple_index(-1, self.n_variables)] = 1

        for degree in range(1, 1 + self.get_degree()):
            coef_indexes = get_coef_indexes(degree, self.n_variables)
            for index in coef_indexes:
                for variable_index in range(self.n_variables):
                    if index[variable_index] != 0:
                        previous_index = list(index)
                        previous_index[variable_index] -= 1
                        previous_index = tuple(previous_index)
                        previous_value = monomial_values[previous_index]
                        value = multiply(
                            previous_value, variables[variable_index])
                        monomial_values[index] = value
                        break

        result = 0
        for coef_index, coef_value in self.coefs.items():
            term = multiply(coef_value, monomial_values[coef_index])
            result = add(result, term)
        return result

    def eval_one_variable(self, var_index: int, value: int):
        """returns the polynomial we get by fixing the value of one variable"""
        assert 0 <= var_index < self.n_variables
        new_polynomial = Polynomial(self.n_variables-1)
        for coef_index, coef_value in self.coefs.items():
            new_coef_index = list(coef_index)
            del new_coef_index[var_index]
            new_coef_index = tuple(new_coef_index)
            new_coef_value = multiply(
                coef_value, power(value, coef_index[var_index]))
            new_polynomial.increase_coef_by(new_coef_index, new_coef_value)
        return new_polynomial

    def get_degree(self) -> int:
        degree = -1
        for index in self.coefs:
            degree = max(degree, sum(index))
        return degree

    def get_linear_coefs(self):
        """Used when the Polynomial has degree 1, that is to say
        when it is a linear combination of the form :
        4x + 5y + 3z + 12
        -> Returns:
        [4, 5, 3, 12]
        """

        assert self.get_degree() <= 1

        coefs_on_list = []
        for coef_number in range(self.n_variables):
            coefs_on_list.append(self.get_coef(
                get_tuple_index(coef_number, self.n_variables)))
        coefs_on_list.append(self.get_coef(
            get_tuple_index(-1, self.n_variables)))  # constant coef

        return coefs_on_list

    def set_alphabet(self, new_alphabet: List[str]):
        """change the symbols used to display the variables of this
        polynomial when __str__(self) is called"""
        self.alphabet = new_alphabet

    def __str__(self) -> str:
        assert self.n_variables <= len(self.alphabet)
        polynomial_str = ""
        for coef_index, coef_value in self.coefs.items():
            coef_str = str(coef_value)
            if sum(coef_index) != 0:
                coef_str += '.'
            for var_index in range(self.n_variables):
                if coef_index[var_index] != 0:
                    coef_str += self.alphabet[var_index]
                    if coef_index[var_index] != 1:
                        coef_str += "^" + str(coef_index[var_index])
            polynomial_str += coef_str + " + "
        return polynomial_str[0:-3]


def add_multivariate_polynomials(a: Polynomial, b: Polynomial) -> Polynomial:
    """Returns the sum of 2 multivariate polynomials
    They must have the same number of unknows"""
    assert a.get_number_of_variables() == b.get_number_of_variables()
    c = Polynomial(a.get_number_of_variables())
    for coef_index in set(a.get_coefs().keys()).union(set(b.get_coefs().keys())):
        c.set_coef(coef_index, add(a.get_coef(
            coef_index), b.get_coef(coef_index)))
    return c


def multiply_multivariate_polynomials(a: Polynomial, b: Polynomial) -> Polynomial:
    """Returns the product of 2 multivariate polynomials
    They must have the same number of unknows"""
    assert a.get_number_of_variables() == b.get_number_of_variables()
    c = Polynomial(a.get_number_of_variables())
    for a_coef_index, a_coef_value in a.get_coefs().items():
        for b_coef_index, b_coef_value in b.get_coefs().items():
            # element wise tuple addition
            new_coef_index = tuple(map(sum, zip(a_coef_index, b_coef_index)))
            new_coef_value = multiply(a_coef_value, b_coef_value)
            new_coef_value = add(new_coef_value, c.get_coef(new_coef_index))
            c.set_coef(new_coef_index, new_coef_value)
    return c


def power_multivariate_polynom(a: Polynomial, n: int) -> Polynomial:
    """Returns the polynomial in parameter, raised to the power n"""
    assert n >= 0
    c = Polynomial(a.get_number_of_variables())
    c.set_coef(tuple([0 for _ in range(a.get_number_of_variables())]), 1)
    for _ in range(n):
        c = multiply_multivariate_polynomials(c, a)
    return c


def multiply_scalar_multivariate_polynom(alpha: int, a: Polynomial) -> Polynomial:
    """Returns the product of the multivariate polynomial a by the scalar alpha"""
    c = Polynomial(a.get_number_of_variables())
    for coef_index, coef_value in a.get_coefs().items():
        new_value = multiply(coef_value, alpha)
        c.set_coef(coef_index, new_value)
    return c


def compose(univariate_polynomial: Polynomial, multivariate_polynomial: Polynomial) -> Polynomial:
    """returns the composition a∘b = a(b)
    a must contain only one variable"""
    assert univariate_polynomial.get_number_of_variables() == 1
    n_variables = multivariate_polynomial.get_number_of_variables()
    new_multivariate_polynomial = Polynomial(n_variables)
    new_multivariate_polynomial.set_coef(
        tuple([0 for _ in range(n_variables)]), univariate_polynomial.get_coef((0,)))
    for power in range(1, univariate_polynomial.get_degree() + 1):
        powered_polynomial = power_multivariate_polynom(
            multivariate_polynomial, power)
        powered_polynomial = multiply_scalar_multivariate_polynom(
            univariate_polynomial.get_coef((power,)), powered_polynomial)
        new_multivariate_polynomial = add_multivariate_polynomials(
            new_multivariate_polynomial, powered_polynomial)
    return new_multivariate_polynomial


def solve_linear_system(linear_system: List[Polynomial]) -> List:
    """Linear system solver with Gaussian elimination:

    If we want to solve the following system :

    (mod 23)
    2x + 3y = 13
    4x + 9y = 12 
    3x + 5y = 21

    The parameter 'linear_system' must be : 
    [Polynomial(2x + 3y - 13), Polynomial(4x + 9y - 44), Polynomial(3x + 5y - 26)] 

    -> returns the list of all the solutions, each solution beeing
    represented by a list of 'number of variables' integers

    In this example : [[2, 3]] is returned because x=2, y=3 is the only solution.

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
                get_tuple_index(var_index, n_variables))
            if pivot != 0:
                # invert lines
                temp = linear_system[line_index]
                linear_system[line_index] = linear_system[var_index]
                linear_system[var_index] = temp
            line_index += 1
        if pivot != 0:
            inv = inverse(pivot)
            linear_system[var_index] = multiply_scalar_multivariate_polynom(
                inv, linear_system[var_index])
            for line_index_2 in range(var_index+1, n_lines):
                alpha = multiply(-1, linear_system[line_index_2].get_coef(
                    get_tuple_index(var_index, n_variables)))
                temp = multiply_scalar_multivariate_polynom(
                    alpha, linear_system[var_index])
                linear_system[line_index_2] = add_multivariate_polynomials(
                    linear_system[line_index_2], temp)
        else:
            solutions = []
            for val in range(specs.p):
                # the variable var_index receives the value 'val'
                new_linear_system = [pol.eval_one_variable(
                    var_index, val) for pol in linear_system]
                temp_solutions = solve_linear_system(new_linear_system)
                if temp_solutions == []:
                    break
                for temp_sol in temp_solutions:
                    solutions.append(
                        temp_sol[0:var_index] + [val] + temp_sol[var_index:])
            return solutions

    for h in range(n_variables, n_lines):
        if linear_system[h].get_coef(get_tuple_index(-1, n_variables)) != 0:
            return []  # no solutions

    sol = [0 for _ in range(n_variables)]
    for line_index in range(n_variables-1, -1, -1):
        sum = linear_system[line_index].get_coef(
            get_tuple_index(-1, n_variables))
        for var_index_2 in range(line_index+1, n_variables):
            sum += multiply(sol[var_index_2], linear_system[line_index].get_coef(
                get_tuple_index(var_index_2, n_variables)))
        sol[line_index] = multiply(-1, sum)
    return [sol]


def solve_linear_system_eq(linear_system: List[Polynomial], values_eq: List[int]) -> List:
    """Similar to 'solve_linear_system' (see just above). Difference : here
    we can have constant terms both at left and right of the linear equations: 

    If we want to solve the following system :

    (mod 23)
    2x + 3y + 7 = 6
    4x + 9y + 1 = 11
    3x + 5y + 9 = 21

    The parameters must be: 
    linear_system = [Polynomial(2x + 3y + 7), Polynomial(4x + 9y + 1), Polynomial(3x + 5y + 76)] 
    values_eq = [4, 5, 26]

    -> returns the list of all the solutions, each solution beeing
    represented by a list of 'number of variables' integers

    In this example : [[5, 4]] is returned because x=5, y=4 is the only solution.

    (Returns [] if there is no solutions)
    """
    assert len(linear_system) == len(values_eq)

    new_linear_system = []
    for index, linear_combination in enumerate(linear_system):
        new_linear_combination = deepcopy(linear_combination)
        tuple_index = get_tuple_index(-1,
                                      linear_combination.get_number_of_variables())
        new_linear_combination.increase_coef_by(
            tuple_index, multiply(-1, values_eq[index]))
        new_linear_system.append(new_linear_combination)
    return solve_linear_system(new_linear_system)


def square_root(a):
    """Implementation of the Tonelli-Shanks algorithm.

    code inspired by : 
    https://gist.github.com/nakov/60d62bdf4067ea72b7832ce9f71ae079

    Fnds x such as x^2 = a (modulo p)
    Returns [x, p-x] the two solutions
    [None, None] is returned is no square root exists.
    """
    def legendre_symbol(a):
        ls = pow(a, (specs.p - 1) // 2, specs.p)
        return -1 if ls == specs.p - 1 else ls

    def enhance_return(one_squarere_root):
        if one_squarere_root is None:
            return [None, None]

        one_squarere_root = one_squarere_root % specs.p
        if one_squarere_root == 0:
            return [0, 0]
        assert 0 <= one_squarere_root < specs.p
        return [one_squarere_root, specs.p - one_squarere_root]

    a = a % specs.p

    if a == 0:
        return enhance_return(0)
    elif legendre_symbol(a) != 1:
        return enhance_return(None)
    elif specs.p == 2:
        return [a, a]
    elif specs.p % 4 == 3:
        return enhance_return(pow(a, (specs.p + 1) // 4, specs.p))
    s = specs.p - 1
    e = 0
    while s % 2 == 0:
        s //= 2
        e += 1
    n = 2
    while legendre_symbol(n) != -1:
        n += 1
    x = pow(a, (s + 1) // 2, specs.p)
    b = pow(a, s, specs.p)
    g = pow(n, s, specs.p)
    r = e

    while True:
        t = b
        m = 0
        for m in range(r):
            if t == 1:
                break
            t = pow(t, 2, specs.p)
        if m == 0:
            return enhance_return(x)
        gs = pow(g, 2 ** (r - m - 1), specs.p)
        g = (gs * gs) % specs.p
        x = (x * gs) % specs.p
        b = (b * g) % specs.p
        r = m


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
        b = inverse(a)*b
        c = inverse(a)*c
        d = inverse(a)*d
        e = substract(d, c)
        f = divide(b, 2)
        g = add(e, power(f, 2))
        h = square_root(g)
        if h[0] is None:
            return []
        else:
            return [substract(h[0], f), substract(h[1], f)]


def compute_groebner_basis(sympy_polynomials):
    grobner_basis = groebner(
        sympy_polynomials, order='lex', domain=FiniteField(specs.p))
    return [pol.expr for pol in grobner_basis.polys]
