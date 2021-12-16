import random
from my_maths import Vector

import specs
from protocol import (
    generate_random_secret_key,
    generate_public_key,
    encrypt,
    decrypt,
    generate_system_needed_to_break_secret_key_security,
)


"""CODE EXAMPLES

We use here "dimensions" to refer to the list [a_1, .. a_m] in the article

"""


def generate_random_message(n_variables: int) -> Vector:
    message = Vector(
        coefs=[random.randint(0, specs.p - 1) for _ in range(n_variables)]
    )
    return message


def example_0() -> None:
    """first test:
    We encrypt and decrypt a simple message.
    For more readability, we temporary use a small value for p here.
    """
    old_p = specs.p
    specs.p = 97  # small p -> low security, but more readability in terminal
    dimensions = [3, 4, 4, 4]

    secret_key = generate_random_secret_key(dimensions)
    print(secret_key)

    public_key = generate_public_key(secret_key,  verbose=False)
    print(public_key)

    message = Vector(coefs=[18, 65, 53])
    print("MESSAGE : " + str(message) + "\n\n")

    encrypted_message = encrypt(message, public_key)
    print("ENCRYPTED MESSAGE : " + str(encrypted_message) + "\n\n")

    decrypted_message = decrypt(encrypted_message, secret_key)
    print("DECRYPTED MESSAGE : " + str(decrypted_message) + "\n\n")

    specs.p = old_p


def example_1() -> None:
    """loop over multiple message to check that the protocol works."""
    dimensions = [5, 6, 7, 8]
    n_tests = 100

    secret_key = generate_random_secret_key(dimensions)
    public_key = generate_public_key(secret_key,  verbose=True)
    print("checking that the protocol works properly... ")
    for test_index in range(n_tests):
        message = generate_random_message(dimensions[0])
        encrypted_message = encrypt(message, public_key)
        potential_decrypted_messages = decrypt(encrypted_message, secret_key)
        assert message in potential_decrypted_messages
        print("test " + str(test_index + 1) + ": encryption/decryption works")
    print("\n\n")


def example_2() -> None:
    dimensions = [3, 3, 4, 4]
    system = generate_system_needed_to_break_secret_key_security(dimensions)
    print("With [a_1, ..., a_m] = " + str(dimensions) + ",")
    print(
        """The system you need to break to recover the secret key from a
    public key is:"""
    )
    for polynomial in system:
        print(str(polynomial) + " = 0\n")
    print(
        (
            "It contains "
            + str(len(system))
            + " multivariate polynomials and "
            + str(system[0].get_number_of_variables())
            + " unknowns."
        )
    )


example_0()
