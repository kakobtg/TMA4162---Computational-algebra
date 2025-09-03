import numpy as np
import time
import matplotlib.pyplot as plt
import random
from concurrent.futures import ThreadPoolExecutor



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


    try:
        max_exp = exp.bit_length() - 1
    except AttributeError as e:
        print(type(exp))
        raise e
        
    precomputed = precompute_exponents(base, max_exp, p)

    result = 1
    base_exp = 1
    
    while exp > 0:
        if exp & 1:  # Check if the least significant bit of exp is 1
            result = modular_multiply(result, precomputed[base_exp], p)
        exp >>= 1
        base_exp <<= 1  # Move to the next precomputed power

    return result


def measure_runtime(function, base, exp, p, num_trials=10):
 # Measures average runtime
    total_time = 0
    for _ in range(num_trials):
        start_time = time.perf_counter()
        function(base, exp, p)
        total_time += time.perf_counter() - start_time
    return total_time / num_trials


def plot_runtimes():
    primes = [10007,11927,101027,1907981,49079827,290802131,500804189,790797121,990801067,2907974869]  # chosen sequence of primes
    naive_times = []
    square_and_multiply_times = []
    windowed_times = []
    
    for p in primes:
        base = random.randint(2, p - 1)
        exp = random.randint(1, p - 1)
        
        naive_time = measure_runtime(naive_exponentiation, base, exp, p)
        square_time = measure_runtime(square_and_multiply, base, exp, p)
        windowed_time = measure_runtime(sliding_window_exponentiation, base, exp, p)
        
        naive_times.append(naive_time)
        square_and_multiply_times.append(square_time)
        windowed_times.append(windowed_time)
        
    
    plt.plot(primes, naive_times, label="Naive Exponentiation")
    plt.plot(primes, square_and_multiply_times, label="Square-and-Multiply")
    plt.plot(primes, windowed_times, label="Windowed exponentiation")
    plt.yscale('log')
    plt.xlabel("Prime Modulus (p)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime Comparison of Exponentiation Algorithms")
    plt.legend()
    plt.grid(True)
    plt.show()



def test_modular():
        
    p = 17 

    assert modular_add(13, 5, p) == 1 
    assert modular_sub(13, 5, p) == 8  
    assert modular_multiply(13, 5, p) == 14 
    assert modular_inverse(13, p) == 4 
    assert modular_divide(13, 5, p) == 6  
    assert naive_exponentiation(7, 8, p) == 16
    assert square_and_multiply(7, 8, p) == 16
    assert sliding_window_exponentiation(7, 8, p) == 16
    
    assert naive_exponentiation(75, 37, p) == 11
    assert square_and_multiply(75, 37, p) == 11
    assert sliding_window_exponentiation(75, 37, p) == 11
    
    
    print("Success")

if __name__ == "__main__":
    test_modular()
    #plot_runtimes()