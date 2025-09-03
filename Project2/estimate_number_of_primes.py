import numpy as np
from matplotlib import pyplot as plt
from sympy import primepi, primerange, prime


def expected_tests_random_search(n: int) -> float:
    return round(n * np.log(2), 2)

def failure_prob(n, p) -> float:
    """
    Trials needed for a geo dist
    failure prob: p = (1 - (1/n ln2)
    ==> log(p) = x log(1 - (1/(n log2)))
        x = log(p) / log(1 - (1/(n log2)))
    """
    return round(np.log(p) / (np.log(1 - (1/(n* np.log(2))))))


def range_for_random_search(n: int, p_success: float) -> float:
    return float(round(-n*np.log(2) * np.log(1 - p_success)))

def sieve_fraction(P: int) -> float:
    # Computes the fraction of numbers not divisible by primes <= P
    primes = list(primerange(2, P+1))
    fraction = np.prod([(1 - 1/p) for p in primes])
    return fraction

def get_primes(count: int) -> list:
    return list(primerange(2, prime(count + 1)))

def expected_tests_filtered(n: int, num_small_primes: list) -> list:
    # Expected number of candidates to test after filtering by small primes
    small_primes = get_primes(num_small_primes)
    
    probability_keep = np.prod([1 - 1 / p for p in small_primes])
    expected_tests = round(n * np.log(2) / probability_keep, 2)
    
    if probability_keep <= 0:  # Avoid division by zero or negative probabilities
        return float('inf')
    
    return float(expected_tests)



if __name__ == "__main__":
    exTest = []
    numTests = []
    probabilities = [0.5, 0.05, 0.01, 0.001]
    nums = [256, 512, 1024, 2048, 4096]
    small_primes = [2,10,50,250,1250,6250,31250]
    for n in nums:  
        for p in probabilities:
            exTest.append(failure_prob(n, p))
            
    matrix = [[expected_tests_filtered(n,k) for k in small_primes] for n in nums]
    
    #filtered_test = [expected_tests_filtered(n, small_primes) for n in nums]
    
    print(exTest)
    print(50*'#')
    print(matrix)
    


        