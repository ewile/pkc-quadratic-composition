# Public key cryptography with polynomial composition

Implementation of a new scheme for public key cryptography.
<br>This code is oriented towards education and research. It is a tool to facilitate further research on the security of this protocol.
<br>Paper (preprint): https://eprint.iacr.org/2021/1476/20211217:102712 <br> (This is the second version of the protocol, the first version was broken. More details can be found in the paper).

## Quick start
```
git clone https://github.com/mi10e3/pkc-quadratic-composition
```

The protocol works modulo p, stored in `specs.py`.

## Example of a simple encryption / decryption

See `example_0` from `test.py` for full details. We use a small value for p (p = 97) for readability.

As explained in the article, another parameter of this protocol is the list [a_1, ..., a_m]. In the code, this list is called `dimensions`:
```python
dimensions = [3, 4, 4, 4]
```

Alice generates her secret key:
```python
secret_key = generate_random_secret_key(dimensions)
print(secret_key)
```

*transformation 0:<br><br>
17 + 67.A + 49.B + 67.C<br>
51 + 71.A + 15.B + 1.C<br>
66 + 49.A + 49.B + 68.C<br>
69 + 24.A + 63.B + 24.C<br><br>
transformation 1:<br><br>
87 + 18.X + 6.X^2<br>
88 + 19.X + 16.X^2<br>
0 + 67.X + 85.X^2<br>
45 + 41.X + 54.X^2<br><br>
transformation 2:<br><br>
63 + 10.A + 48.B + 91.C + 65.D<br>
11 + 26.A + 58.B + 66.C + 33.D<br>
47 + 27.A + 96.B + 78.C + 61.D<br>
35 + 80.A + 2.B + 82.C + 89.D<br><br>
transformation 3:<br><br>
93 + 63.X + 71.X^2<br>
2 + 9.X + 29.X^2<br>
68 + 89.X + 36.X^2<br>
96 + 33.X + 51.X^2<br><br>
transformation 4:<br><br>
51 + 62.A + 40.B + 34.C + 12.D<br>
32 + 59.A + 25.B + 44.C + 17.D<br>
55 + 14.A + 0.B + 58.C + 76.D<br>
58 + 20.A + 14.B + 1.C + 93.D*

Alice generates the resulting public key:
```python
public_key = generate_public_key(secret_key)
print(public_key)

```

*71.A^2C^2 + 12.B + 56.A^2B + 47.BC^3 + 90.B^3 + 72.AB^3 + 4.C + 40.C^4 + 43.B^2C + 61.<span>AC + 30.AB + 2.A^3 + 13.A^4 + 69.A^2C + 24.BC^2 + 88.AB^2C + 76.A^3C + 54.C^3 + 77.B^2 + 67 + 20.B^4 + 94.ABC^2 + 17.A + 18.<span>AC^3 + 57.A^2 + 87.A^2B^2 + 80.BC + 52.A^2BC + 27.B^3C + 5.AB^2 + 64.A^3B + 7.C^2 + 93.B^2C^2 + 87.AC^2 + 28.ABC
<br><br>
88.A^2C^2 + 38.B + 58.A^2B + 85.BC^3 + 66.B^3 + 9.AB^3 + 61.C + 64.C^4 + 9.B^2C + 11.<span>AC + 64.AB + 55.A^3 + 7.A^4 + 48.A^2C + 43.BC^2 + 29.AB^2C + 52.A^3C + 49.C^3 + 19.B^2 + 25 + 44.B^4 + 90.ABC^2 + 57.A + 8.AC^3 + 94.A^2 + 31.A^2B^2 + 75.BC + 91.A^2BC + 1.B^3C + 33.AB^2 + 25.A^3B + 77.C^2 + 53.B^2C^2 + 80.<span>AC^2 + 8.ABC
<br><br>
47.A^2C^2 + 24.B + 94.A^2B + 33.BC^3 + 13.B^3 + 47.AB^3 + 12.C + 84.C^4 + 16.B^2C + 57.<span>AC + 63.AB + 46.A^3 + 90.A^4 + 76.A^2C + 33.BC^2 + 89.AB^2C + 7.A^3C + 8.C^3 + 8.B^2 + 49 + 12.B^4 + 90.ABC^2 + 58.A + 20.<span>AC^3 + 25.A^2 + 80.A^2B^2 + 0.BC + 44.A^2BC + 80.B^3C + 3.AB^2 + 41.A^3B + 77.C^2 + 24.B^2C^2 + 32.AC^2 + 52.ABC
<br><br>
29.A^2C^2 + 56.B + 8.A^2B + 52.BC^3 + 54.B^3 + 37.AB^3 + 11.C + 89.C^4 + 82.B^2C + 7.<span>AC + 86.AB + 79.A^3 + 46.A^4 + 76.A^2C + 79.BC^2 + 3.AB^2C + 13.A^3C + 21.C^3 + 67.B^2 + 40 + 15.B^4 + 37.ABC^2 + 53.A + 81.<span>AC^3 + 76.A^2 + 74.A^2B^2 + 14.BC + 13.A^2BC + 50.B^3C + 12.AB^2 + 64.A^3B + 82.C^2 + 7.B^2C^2 + 29.<span>AC^2 + 67.ABC*


Bob wants to send a message to Alice:

```python
message = Vector(coefs=[18, 65, 53])
encrypted_message = encrypt(message, public_key)
print(encrypted_message)
```

*ENCRYPTED MESSAGE: [93, 49, 49, 28]*

```python
decrypted_message = decrypt(encrypted_message, secret_key)
print(decrypted_message)
```

*DECRYPTED MESSAGE: [[18, 65, 53]]*

Other examples can be found in `test.py` For instance, `example_2` generates the polynomial system that guarantees the security of the keys (the system of multivariate polynomial we need to solve to recover the secret key from the public key).


## Author

Emile Hautefeuille
