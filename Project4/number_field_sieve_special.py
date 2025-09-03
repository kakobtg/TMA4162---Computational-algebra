import numpy as np
from sympy import Matrix, gcd, nextprime, isprime, primerange
from sympy.ntheory import factorint
import math

# Helper method to choose optimal parameters
def choose_optimal_res(n):
    # heuristic search for best special form n = r^e - s
    optimal = (None, None, None)
    min_cost = float('inf')

    max_e = int(math.log2(n)) + 1
    for e in range(2, max_e):
        r = int(n ** (1 / e))
        for candidate_r in [r, r + 1]:
            for s in [-1, 1]:
                if candidate_r**e + s == n:
                    cost = candidate_r + e  # simplistic cost heuristic: minimize r and maximize e
                    if cost < min_cost:
                        min_cost = cost
                        optimal = (candidate_r, e, s)

    if optimal == (None, None, None):
        raise ValueError("Could not find optimal parameters r, e, s")

    return optimal


# Step 1: Polynomial Selection (Generalized)
def select_polynomial(r, e, s):
    d = max(2, int(np.cbrt(e)))  # heuristic choice for d
    k = int(np.ceil(e / d))
    t = -s * r ** (k * d - e)
    f_coeffs = [0] * (d + 1)
    f_coeffs[0] = -t
    f_coeffs[-1] = 1
    return f_coeffs, r ** k

# Explicit norm computation for algebraic numbers
def algebraic_norm(f_coeffs, a, b):
    d = len(f_coeffs) - 1
    return abs(sum(f_coeffs[i] * a**(d - i) * b**i for i in range(d + 1)))

# Step 2: Factor Base Construction for rational and algebraic sides
def construct_factor_bases(B, f_coeffs):
    rational_base = list(primerange(2, B))
    algebraic_base = []
    for p in rational_base:
        for c in range(p):
            poly_val = sum(f_coeffs[i] * c ** i for i in range(len(f_coeffs)))
            if poly_val % p == 0:
                algebraic_base.append(p)
                break
    return rational_base, algebraic_base

# Smoothness check explicitly defined
def is_smooth(num, base):
    factors = factorint(abs(num))
    return all(p in base for p in factors)

# Step 3: Two-step Sieving explicitly performed
def sieve_smooth_pairs(f_coeffs, m, rational_base, algebraic_base, sieve_limit):
    smooth_pairs = []
    for b in range(1, sieve_limit + 1):
        for a in range(-sieve_limit, sieve_limit + 1):
            rational_val = a + b * m
            algebraic_val = algebraic_norm(f_coeffs, a, b)
            if gcd(a, b) == 1 and is_smooth(rational_val, rational_base) and is_smooth(algebraic_val, algebraic_base):
                smooth_pairs.append((a, b))
            if len(smooth_pairs) > len(rational_base) + len(algebraic_base):
                print(f"Sieving stopped early with {len(smooth_pairs)} smooth pairs.")  # Debugging
                return smooth_pairs
            
    print(f"Sieving completed with {len(smooth_pairs)} smooth pairs.")  # Debugging
    return smooth_pairs

# Step 4: Linear Algebra using Gaussian elimination explicitly
# Matrix built mod 2 from rational and algebraic smooth numbers
def build_matrix(smooth_pairs, rational_base, algebraic_base, f_coeffs, m):
    matrix = []
    for a, b in smooth_pairs:
        row = []
        rat_factors = factorint(abs(a + b * m))
        alg_factors = factorint(algebraic_norm(f_coeffs, a, b))
        for p in rational_base:
            row.append(rat_factors.get(p, 0) % 2)
        for p in algebraic_base:
            row.append(alg_factors.get(p, 0) % 2)
        matrix.append(row)
    return Matrix(matrix)

# Step 5: Square roots explicitly computed
def find_square_root_solution(matrix):
    nullspace = matrix.nullspace()
    if nullspace:
        solution = nullspace[0]
        return [i for i, x in enumerate(solution) if x % 2 != 0]
    else:
        raise ValueError("No square root solution found")

# Step 6: Factorization via GCD explicitly performed
def factorize(solution_indices, smooth_pairs, f_coeffs, m, n):
    X, Y = 1, 1
    for idx in solution_indices:
        a, b = smooth_pairs[idx]
        X *= a + b * m
        Y *= algebraic_norm(f_coeffs, a, b)
    X, Y = abs(X), abs(Y)
    factor1 = gcd(X - Y, n)
    factor2 = gcd(X + Y, n)
    if factor1 in (1, n) or factor2 in (1, n):
        raise ValueError("Non-trivial factors not found")
    return factor1, factor2

# Integrated SNFS algorithm using above explicit procedures
def SNFS(n, r, e, s, B=500, sieve_limit=5000):
    f_coeffs, m = select_polynomial(r, e, s)
    rational_base, algebraic_base = construct_factor_bases(B, f_coeffs)
    smooth_pairs = sieve_smooth_pairs(f_coeffs, m, rational_base, algebraic_base, sieve_limit)
    
    print(f"Rational base size: {len(rational_base)}")  # Debugging
    print(f"Algebraic base size: {len(algebraic_base)}") # Debugging
    
    matrix = build_matrix(smooth_pairs, rational_base, algebraic_base, f_coeffs, m)
    print(f"Matrix dimensions: {matrix.rows} x {matrix.cols}")  # Debugging
    
    try:
      solution_indices = find_square_root_solution(matrix)
      return factorize(solution_indices, smooth_pairs, f_coeffs, m, n)
    except ValueError as e:
        print(e)
        return 1, n

# Example usage explicitly illustrating features
if __name__ == '__main__':
    n = 2**104 + 1
    
    # Automated the parameter selection because it it exhausting to wirite more than number per test
    r, e, s = choose_optimal_res(n)
    print(f"Optimal parameters for {n} are r={r}, e={e}, s={s}")
    factor1, factor2 = SNFS(n, r, e, s)
    print(f"SNFS factors of {n} are {factor1} and {factor2}")