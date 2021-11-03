
import random
import time
from matplotlib import pyplot as plt

from math_modulo import compute_groebner_basis
import specs
from protocol import generate_random_secret_key, generate_public_key, encrypt, decrypt
from symbolic import generate_system_needed_to_break_secret_key_security, pretty_print_polyomial_system


def generate_ranndom_message(n_variables):
    message = [random.randint(0, specs.p-1) for _ in range(n_variables)]
    return tuple(message)


def example_0() -> None:
    """educational first test
    we encrypt and decrypt a simple message
    for more readability, we temporary use a small value for p here
    """
    old_p = specs.p
    specs.p = 97  # small p -> low security, but more readability in terminal
    n_lines = 4
    n_variables = 3
    n_compositions = 2

    secret_key = generate_random_secret_key(
        n_lines, n_compositions, n_variables)
    print(secret_key)

    public_key = generate_public_key(secret_key, verbose=True)
    print(public_key)

    message = (7, 53, 11)
    print('MESSAGE : ' + str(message) + '\n\n')

    encrypted_message = encrypt(message, public_key)
    print('ENCRYPTED MESSAGE : ' + str(encrypted_message) + '\n\n')

    decrypted_message = decrypt(encrypted_message, secret_key)
    print('DECRYPTED MESSAGE : ' + str(decrypted_message) + '\n\n')

    specs.p = old_p


def example_1(n_lines: int, n_compositions: int, n_variables: int, n_tests: int) -> None:
    # loop over multiple message to check that the protocol works

    secret_key = generate_random_secret_key(
        n_lines, n_compositions, n_variables)
    public_key = generate_public_key(secret_key, verbose=True)
    print('checking that the protocol works properly... ')
    for test_index in range(n_tests):
        message = generate_ranndom_message(n_variables)
        encrypted_message = encrypt(message, public_key)
        potential_decrypted_messages = decrypt(encrypted_message, secret_key)
        assert message in potential_decrypted_messages
        print('test ' + str(test_index+1) + ': encryption/decryption works')
    print('\n\n')


def example_2(n_lines: int, n_compositions: int, n_variables: int) -> None:
    # The size of the keys (both secret and public) grows
    # linearly with the prime modulo (specs.p). There is
    # also a linear dependency between the prime modulo and
    # the encryption / decryption time, as shown by the following
    # curves :

    old_p = specs.p

    prime_number_examples = specs.prime_number_examples[0:10]

    n_keys_per_prime_number = 10
    n_tests_per_key = 10

    times_to_generate_secret_key = [0.0]*len(prime_number_examples)
    times_to_generate_public_key = [0.0]*len(prime_number_examples)
    times_to_encrypt = [0.0]*len(prime_number_examples)
    times_to_decrypt = [0.0]*len(prime_number_examples)

    print('measuring how changing the prime modulo affects the different times:')
    for index, prime_number in list(enumerate(prime_number_examples)):
        print('test '+str(index)+': using p='+str(prime_number))
        specs.p = prime_number

        for _ in range(n_keys_per_prime_number):
            t1 = time.time()
            secret_key = generate_random_secret_key(
                n_lines, n_compositions, n_variables)
            t2 = time.time()
            times_to_generate_secret_key[index] += t2-t1

            t1 = time.time()
            public_key = generate_public_key(secret_key)
            t2 = time.time()
            times_to_generate_public_key[index] += t2-t1

            for __ in range(n_tests_per_key):
                message = generate_ranndom_message(n_variables)

                t1 = time.time()
                encrypted_message = encrypt(message, public_key)
                t2 = time.time()
                times_to_encrypt[index] += t2-t1

                t1 = time.time()
                potential_decrypted_messages = decrypt(
                    encrypted_message, secret_key)
                t2 = time.time()
                times_to_decrypt[index] += t2-t1

        times_to_encrypt[index] /= n_keys_per_prime_number*n_tests_per_key
        times_to_decrypt[index] /= n_keys_per_prime_number*n_tests_per_key
        times_to_generate_secret_key[index] /= n_keys_per_prime_number
        times_to_generate_public_key[index] /= n_keys_per_prime_number

    times_and_titles = [[times_to_generate_secret_key, 'duration of secret key generation'],
                        [times_to_generate_public_key,
                            'duration of public key generation'],
                        [times_to_encrypt, 'average duration of encryption'],
                        [times_to_decrypt, 'average duration of decryption']]

    for index in range(len(times_and_titles)):
        times_and_titles[index][1] += '\n('+str(n_variables) + ' variables, '+str(
            n_compositions)+' compositions, ' + str(n_lines) + ' lines)'

    specs.p = old_p

    for times, title in times_and_titles:
        X = [len(str(prime)) for prime in prime_number_examples]
        plt.figure()
        plt.plot(X, times, marker='o')
        plt.title(title)
        plt.xlabel(
            'number of digits (base 10) of p, the caracteristic of our finite field')
        plt.ylabel('time (s)')
    plt.show()


def example_3(n_compositions: int, n_variables: int) -> None:
    """The security of this protocol relies on the system
    of multivariate polynomials of the public key. However,
    an attacker could also try to compute back the secret
    key from the public key. This attack would also involve
    solving a system of multivariate polynomials, suppositely
    easier to solve (ie less general). This function computes
    this system, and print it. If someone can solve it, it means
    that the scheme is unsafe with the current parameters
    (specs.p, n_compositions, n_variables). Finally, this function
    tries to compute a groebner basis with sympy."""

    sympy_polynomials = generate_system_needed_to_break_secret_key_security(
        n_compositions, n_variables)

    print("\n\nExample of a system to solve in order to compute back the public key from the secret key:\n(" +
          str(n_compositions) + " compositions, "+str(n_variables)+" variables)\n\n")
    pretty_print_polyomial_system(sympy_polynomials)

    print('\n\nComputing groebner basis ...')
    grobner_polynomials = compute_groebner_basis(sympy_polynomials)

    print("\n\nGroebner basis:\n\n")
    pretty_print_polyomial_system(grobner_polynomials)


example_0()

example_1(n_lines=specs.n_lines,
          n_compositions=specs.n_compositions,
          n_variables=specs.n_variables,
          n_tests=100)

example_2(n_lines=specs.n_lines,
          n_compositions=specs.n_compositions,
          n_variables=specs.n_variables)

example_3(n_compositions=specs.n_compositions,
          n_variables=specs.n_variables)
