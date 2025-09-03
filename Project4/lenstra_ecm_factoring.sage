from sage.all import *
from random import randint
from math import gcd


def lenstra_ecm(N, B, max_tries):
    """
    Try to factor N using Lenstra's Elliptic Curve Method (ECM).

    Args:
        N (int): Composite number to factor.
        B (int): Smoothness bound.
        max_tries (int): Number of random curves to try.

    Returns:
        A nontrivial factor of N if found, otherwise None.
    """
    for attempt in range(max_tries):
        # Pick random curve: y^2 = x^3 + a*x + b mod N
        x0 = randint(1, N - 1)
        y0 = randint(1, N - 1)
        a = randint(1, N - 1)
        b = (y0^2 - x0^3 - a * x0) % N

        try:
            E = EllipticCurve(Integers(N), [a, b])
            P = E(x0, y0)
        except (TypeError, ValueError):
            continue  # invalid point or curve

        # Compute k = product of small prime powers up to B
        k = 1
        for p in prime_range(2, B + 1):
            e = floor(log(B) / log(p))
            k *= p^e

        print(f"Trying curve {attempt + 1}: y^2 = x^3 + {a}x + {b}, point = ({x0}, {y0})")

        try:
            Q = k * P
        except ZeroDivisionError as e:
            # If division fails inside EC arithmetic, extract GCD
            num = e.args[0]
            factor = gcd(num, N)
            if 1 < factor < N:
                print(f"Nontrivial factor found: {factor}")
                return factor
            else:
                continue

    print("No factor found after trying all curves.")
    return None

def lenstra_ecm_fast(N, B, max_tries):
    primes = list(prime_range(2, B + 1))
    k = 1
    for p in primes:
        e = floor(log(B) / log(p))
        k *= p**e

    for attempt in range(max_tries):
        while True:
            x0 = randint(1, N - 1)
            y0 = randint(1, N - 1)
            a = randint(1, N - 1)
            b = (y0^2 - x0^3 - a * x0) % N

            # Compute discriminant
            Delta = (-16) * (4 * a^3 + 27 * b^2) % N
            g = gcd(Delta, N)

            if 1 < g < N:
                print(f"✅ Found factor from discriminant: {g}")
                return g

            if g != 1:
                continue  # bad discriminant, retry

            try:
                E = EllipticCurve(Integers(N), [a, b])
                P = E(x0, y0)
                break
            except (TypeError, ValueError, ArithmeticError):
                continue  # invalid curve or point, retry

        try:
            Q = k * P
        except ZeroDivisionError:
            print(f"⚠ ZeroDivisionError on attempt {attempt + 1} — skipping curve.")
            continue

    print("❌ No factor found after all attempts.")
    return None

if __name__ == "__main__":
    # Example usage
    #N = 1238926361552897 * 93461639715357977769163558199606896584051237541638188580280321  # F_8 Fermat number
    # factor = lenstra_ecm(N, B=1000, max_tries=50)
    # print(f"Found factor: {factor}")

    # Test with a smaller composite number
    #N = 595945676543  # Example: 59 * 101
    p = random_prime(2^15, lbound=2^14)
    q = random_prime(2^15, lbound=2^14)
    N = p * q
    #fac = lenstra_ecm(N, B=5000, max_tries=15000)
    factor = lenstra_ecm_fast(N, B=5000, max_tries=15000)
    if factor:
        print(f"Final result: {factor} * {N // factor} = {N}")  # Should print the factor and its cofactor
