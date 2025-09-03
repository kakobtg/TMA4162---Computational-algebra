import math
import sympy
import numpy as np
from collections import defaultdict

def polynomial_selection(N):
    degree = 3 if N < 10**6 else 5  # Use degree-3 for smaller numbers
    coeffs = [1] + [0] * (degree - 1) + [-N]  # Example polynomial: x^3 - N or x^5 - N
    return np.poly1d(coeffs)

def sieve(N, B):
    # Improved sieve with modular conditions
    smooth_relations = []
    for a in range(1, B * 2):
        for b in range(1, B * 2):
            if math.gcd(a, b) != 1:
                continue
            val = a - b * N
            factors = sympy.factorint(abs(val))
            if all(p <= B for p in factors):
                smooth_relations.append((a, b, factors))
    return smooth_relations

def build_matrix(relations, B):
    primes = list(sympy.primerange(2, B))
    matrix = []

    for _, _, factors in relations:
        row = [0] * len(primes)
        for prime in factors:
            if prime in primes:
                row[primes.index(prime)] = factors[prime] % 2
        matrix.append(row)

    return np.array(matrix, dtype=int), primes

def solve_matrix(matrix):
    # Gaussian elimination for finding linear dependencies
    _, _, V = np.linalg.svd(matrix)
    null_space = V[-1]  # Last row corresponds to the null space vector
    return null_space

def reconstruct_factors(relations, null_space, N):
    x = 1
    y = 1
    for i, coef in enumerate(null_space):
        if coef != 0:  # Non-zero entries
            a, b, _ = relations[i]
            x *= a - b * N
            y *= a
    x, y = abs(x), abs(y)
    gcd = math.gcd(x - y, N)
    if gcd != 1 and gcd != N:
        return gcd
    return None

def gnfs(N, B=300):  # Increased bound to improve success rate
    print(f"Factoring {N} using GNFS with bound {B}.")

    poly = polynomial_selection(N)
    print(f"Selected polynomial: {poly}")

    relations = sieve(N, B)
    prime_list = list(sympy.primerange(2, B))  # Fix for generator issue
    if len(relations) < len(prime_list):
        print("Insufficient smooth relations. Increase bound.")
        return None

    matrix, primes = build_matrix(relations, B)
    null_space = solve_matrix(matrix)
    factor = reconstruct_factors(relations, null_space, N)

    if factor:
        print(f"Non-trivial factor found: {factor}")
        return factor
    else:
        print("Factorization failed. Try adjusting parameters.")
        return None

# Example Usage
N = 87463  # Example composite number
gnfs(N, B=300)