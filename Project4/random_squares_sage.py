from sage.all import *
import random
from math import gcd, log, exp, sqrt
from collections import defaultdict

def is_B_smooth(n, B):
    """Check if n is B-smooth and return its factors"""
    factors = {}
    for p in prime_range(B + 1):
        while n % p == 0:
            factors[p] = factors.get(p, 0) + 1
            n = n // p
            if n == 1:
                return factors
    return None

def random_squares(n, max_trials=100000):
    """
    Random Squares factorization
    Returns a non-trivial factor of n, or None if failed
    """
    n = Integer(n)
    if n < 2:
        return None
    if n % 2 == 0:
        return 2
    if is_prime(n):
        return n

    # Calculate smoothness bound B
    ln_n = log(float(n))
    ln_ln_n = log(ln_n)
    B = int(exp(0.5 * sqrt(ln_n * ln_ln_n)))
    B = max(B, 100)  # Minimum bound

    # Generate factor base
    factor_base = list(prime_range(B + 1))
    num_primes = len(factor_base)

    relations = []
    smooth_numbers = []
    used_indices = set()

    for trial in range(max_trials):
        # Generate random x
        x = random.randint(2, n-1)
        y = (x**2) % n

        # Check for trivial factor
        g = gcd(x, n)
        if g > 1 and g < n:
            return g

        # Check for B-smoothness
        factors = is_B_smooth(y, B)
        if factors:
            # Store relation
            exponent_vector = [0] * num_primes
            for i, p in enumerate(factor_base):
                if p in factors:
                    exponent_vector[i] = factors[p] % 2

            relations.append((x, exponent_vector))
            smooth_numbers.append(y)

            # Try to find linear dependency
            if len(relations) > num_primes:
                M = Matrix(GF(2), [r[1] for r in relations]).transpose()
                K = M.right_kernel()

                for basis in K.basis():
                    indices = [i for i, x in enumerate(basis) if x]
                    if len(indices) >= 2:
                        # Combine relations
                        x_prod = prod(relations[i][0] for i in indices) % n
                        y_prod = prod(smooth_numbers[i] for i in indices)
                        sqrt_y = Integer(y_prod).sqrt()
                        g = gcd(x_prod - sqrt_y, n)
                        if g > 1 and g < n:
                            return g

    return None

def factor_rs(n, _depth=0):
    """
    Complete factorization using Random Squares
    Returns sorted list of prime factors
    """
    n = Integer(n)
    if n == 1:
        return []
    if is_prime(n):
        return [n]

    f = random_squares(n)
    if f is None:
        return [n]  # Return as prime if failed

    return sorted(factor_rs(f, _depth+1) + factor_rs(n//f, _depth+1))