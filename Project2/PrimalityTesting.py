import numpy as np
import time
import matplotlib.pyplot as plt
import random


def gcd_extended(a: int, b: int) -> int:
    if a == 0:
        return b, 0, 1
    gcd, x1, y1 = gcd_extended(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    
    # Returns gcd, x, y such that ax + by = gcd(a,b)
    return gcd, x, y

def modular_inverse(a: int, p: int) -> int:
    gcd, x, _ = gcd_extended(a, p)
        
    if gcd != 1:
        raise ValueError(f'Modular inverse does not exist')
    return x % p

def modular_add(a: int, b: int, p: int) -> int:
    return (a + b) % p

def modular_sub(a: int, b: int, p: int) -> int:
    return (a-b) % p

def modular_multiply(a: int, b: int, p: int) -> int:
    return (a*b) % p

def modular_divide(a: int, b: int, p: int) -> int:
    b_inv = modular_inverse(b, p)
    return modular_multiply(a, b_inv, p)

def naive_exponentiation(base: int, exp: int, p: int) -> int:
    result = 1
    
    for _ in range(exp):
        result = modular_multiply(result, base, p)
    return result

def square_and_multiply(base: int, exp: int, p: int) -> int:
    result = 1
    
    base = base % p
    
    while exp > 0:
        if exp % 2 == 1:
            result = modular_multiply(result, base, p)
        base = modular_multiply(base, base, p)
        exp //= 2
    
    return result

def precompute_exponents(base: int, max_exp: int, p: int) -> int:
    precomputed = {}
    current = base % p
    for i in range(max_exp + 1):
        precomputed[1 << i] = current
        current = modular_multiply(current, current, p)
    return precomputed

def sliding_window_exponentiation(base: int, exp: int, p: int) -> int:
    if exp == 0:
        return 1

    max_exp = exp.bit_length() - 1
    precomputed = precompute_exponents(base, max_exp, p)

    result = 1
    base_exp = 1
    
    while exp > 0:
        if exp & 1:  # Check if the least significant bit of exp is 1
            result = modular_multiply(result, precomputed[base_exp], p)
        exp >>= 1
        base_exp <<= 1  # Move to the next precomputed power

    return result

def fermat_primality_test(n, k=5):
    # True == MaybePrime
    # False == Composite
    if n <= 1:
        return False
    if n <= 3:
        return True

    for _ in range(k):
        a = random.randint(2, n - 2)
        if square_and_multiply(a, n - 1, n) != 1:
            return False
    return True

def miller_rabin_primality_test(n, k):
    # True == MaybePrime
    # False == Composite
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return False

    # Write n-1 as 2^s * d
    s, d = 0, n - 1
    while d % 2 == 0:
        s += 1
        d //= 2

    for _ in range(k):
        a = random.randint(2, n - 2)
        x = square_and_multiply(a, d, n)
        if x == 1 or x == n - 1:
            continue

        for _ in range(s - 1):
            x = modular_multiply(x, x, n)
            if x == n - 1:
                break
        else:
            return False

    return True


def generate_small_primes(limit):
    primes = []
    sieve = [True] * (limit + 1)
    for num in range(2, limit + 1):
        if sieve[num]:
            primes.append(num)
            for multiple in range(num * num, limit + 1, num):
                sieve[multiple] = False
    return primes

def find_prime_in_range(Sn, primality_test):
    attempts = 0
    while True:
        candidate = random.randint(Sn[0], Sn[1])
        attempts += 1
        if primality_test(candidate):
            return candidate, attempts

def find_prime_with_trial_division(Sn, small_primes, primality_test):
    attempts = 0
    while True:
        candidate = random.randint(Sn[0], Sn[1])
        attempts += 1
        if any(candidate % p == 0 for p in small_primes if p < candidate):
            continue
        if primality_test(candidate):
            return candidate, attempts

def find_prime_with_sieving(Sn, sieve_limit, primality_test):
    small_primes = generate_small_primes(sieve_limit)
    start, end = Sn
    attempts = 0
    
    while True:
        candidate = random.randint(start, end)
        attempts += 1
        if all(candidate % p != 0 for p in small_primes):
            if primality_test(candidate):
                return candidate, attempts

def experiment_prime_finding():
    bit_lengths = [512, 1024, 2048, 4096]
    small_prime_counts = [10, 50, 100, 1000, 10000]
    primality_test = miller_rabin_primality_test  # Define this function separately
    results = []
    
    for bits in bit_lengths:
        Sn = (2**bits, 2**(bits+1) - 1)
        
        # Run find_prime_in_range once
        start_time = time.perf_counter()
        prime, attempts = find_prime_in_range(Sn, primality_test)
        elapsed_time = time.perf_counter() - start_time
        print(f"Random Selection (bits={bits}): Found {prime} in {elapsed_time:.6f} sec, {attempts} candidates tested")
        
        for small_prime_limit in small_prime_counts:
            small_primes = generate_small_primes(small_prime_limit)
            
            for name, method in [
                ("Trial Division", find_prime_with_trial_division),
                ("Sieving", find_prime_with_sieving)
            ]:
                start_time = time.perf_counter()
                prime, attempts = method(Sn, small_primes, primality_test) if name == "Trial Division" else method(Sn, small_prime_limit, primality_test)
                elapsed_time = time.perf_counter() - start_time
                results.append((bits, small_prime_limit, name, elapsed_time, attempts))
                print(f"{name} (bits={bits}, small_primes={small_prime_limit}): Found {prime} in {elapsed_time:.6f} sec, {attempts} candidates tested")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    for name in ["Trial Division", "Sieving"]:
        for bits in bit_lengths:
            subset = [(sp, time) for b, sp, n, time, _ in results if n == name and b == bits]
            sp_values, times = zip(*subset)
            plt.plot(sp_values, times, marker='o', label=f"{name}, {bits} bits")
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Number of Small Primes")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.title("Execution Time vs. Small Prime Count")
    plt.show()

def test_primality():
    
     # Fermat Primality Test
    assert fermat_primality_test(1907981) is True
    assert fermat_primality_test(1907983) is False

    # Miller-Rabin Primality Test
    assert miller_rabin_primality_test(1907981) is True
    assert miller_rabin_primality_test(1907982) is False

    
    print("Success")

if __name__ == "__main__":
    test_primality()
    experiment_prime_finding()