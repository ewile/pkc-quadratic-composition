# Public key cryptography with polynomial composition

Implementation of a new scheme for public key cryptography.
<br>This code is educational and research oriented. It is a tool to facilitate further investigation on the security of this protocol.
<br>Paper (preprint): https://eprint.iacr.org/...

## Quick start
```
git clone https://github.com/mi10e3/pkc-quadratic-composition
```
```
pip install -r requirements.txt
```

As explained in the paper, this protocol has 4 parameters :  
- p (a big prime number)
- n_lines
- n_compositions
- n_variables

They can be changed in `specs.py`.

## Examples

Different samples of code can be found in `test.py`.

### Simple encryption / decryption

See example_0 from `test.py` for full details. We use a small value for p (p = 73) for readability.


```python
secret_key = generate_random_secret_key(n_lines=4, n_compositions=2, n_variables=3)
print(secret_key)
```

*(5 + 94.X + 89.X^2) o (21 + 9.X + 57.X^2) o (11 + 56.A + 50.B + 32.C)
<br><br>(65 + 26.X + 91.X^2) o (70 + 6.X + 37.X^2) o (38 + 63.A + 87.B + 78.C)
<br><br>(37 + 53.X + 33.X^2) o (12 + 10.X + 81.X^2) o (90 + 43.A + 70.B + 14.C)
<br><br>(1 + 53.X + 20.X^2) o (65 + 91.X + 75.X^2) o (29 + 90.A + 30.B + 40.C)*

```python
public_key = generate_public_key(secret_key)
print(public_key)
```

*53.A^2C^2 + 56.B + 20.A^2B + 53.BC^3 + 52.B^3 + 17.AB^3 + 63.C + 24.C^4 + 30.B^2C + 78.<span>AC + 37.AB + 1.A^3 + 22.A^4 + 71.A^2C + 58.BC^2 + 21.AB^2C + 78.A^3C + 46.C^3 + 91.B^2 + 21 + 93.B^4 + 60.ABC^2 + 86.A + 71.AC^3 + 44.A^2 + 79.A^2B^2 + 35.BC + 8.A^2BC + 79.B^3C + 4.AB^2 + 37.A^3B + 50.C^2 + 6.B^2C^2 + 96.AC^2 + 9.ABC
<br><br>18.A^2C^2 + 36.B + 21.A^2B + 6.BC^3 + 58.B^3 + 89.AB^3 + 49.C + 95.C^4 + 59.B^2C + 26.<span>AC + 29.AB + 82.A^3 + 44.A^4 + 69.A^2C + 83.BC^2 + 32.AB^2C + 47.A^3C + 17.C^3 + 87.B^2 + 53 + 85.B^4 + 22.ABC^2 + 6.A + 1.AC^3 + 59.A^2 + 95.A^2B^2 + 59.BC + 70.A^2BC + 64.B^3C + 29.AB^2 + 86.A^3B + 90.C^2 + 66.B^2C^2 + 30.AC^2 + 52.ABC
<br><br>66.A^2C^2 + 67.B + 45.A^2B + 41.BC^3 + 28.B^3 + 65.AB^3 + 91.C + 36.C^4 + 75.B^2C + 12.<span>AC + 60.AB + 30.A^3 + 33.A^4 + 9.A^2C + 15.BC^2 + 39.AB^2C + 52.A^3C + 1.C^3 + 15.B^2 + 25 + 93.B^4 + 66.ABC^2 + 37.A + 82.AC^3 + 60.A^2 + 1.A^2B^2 + 6.BC + 78.A^2BC + 55.B^3C + 71.AB^2 + 66.A^3B + 20.C^2 + 65.B^2C^2 + 30.AC^2 + 9.ABC
<br><br>30.A^2C^2 + 44.B + 19.A^2B + 82.BC^3 + 51.B^3 + 28.AB^3 + 91.C + 92.C^4 + 10.B^2C + 31.<span>AC + 96.AB + 19.A^3 + 92.A^4 + 90.A^2C + 78.BC^2 + 15.AB^2C + 45.A^3C + 67.C^3 + 16.B^2 + 93 + 67.B^4 + 20.ABC^2 + 35.A + 52.AC^3 + 47.A^2 + 29.A^2B^2 + 75.BC + 45.A^2BC + 34.B^3C + 71.AB^2 + 58.A^3B + 50.C^2 + 68.B^2C^2 + 40.AC^2 + 60.ABC*

```python
message = (7, 53, 11)

encrypted_message = encrypt(message, public_key)
print(encrypted_message)
```

*(63, 28, 2, 1)*

```python
decrypted_message = decrypt(encrypted_message, secret_key)
print(decrypted_message)
```

*{(7, 53, 11)}*

Other examples can be found in `test.py` For instance, example_3 generates the polynomial system that garantees the security of the secret key (the system of multivariate polynomial we need to solve to compute back the secret key from the public key) and try to compute groebner basis.


## Author

Emile Hautefeuille
