from sage.all import *
import csv
import time
import matplotlib.pyplot as plt
from quadratic_sieve_sage import factor_qs
from mpqs6_sage import mpqs
from random_squares_sage import factor_rs

def load_numbers(filename):
    """Load numbers from first column of CSV"""
    numbers = []
    with open(filename) as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            try:
                n = Integer(eval(row[0].replace(' ', '')))
                numbers.append(n)
            except:
                print(f"Skipping invalid number: {row[0]}")
    return numbers

def load_factors(filename):
    """Load factors from CSV"""
    factors = []
    with open(filename) as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                factors.append((row[1]))
            except:
                print(f"Skipping invalid factors: {row[1]}")
    return factors


def run_factorizations(numbers, algorithm):
    """Run factorization and prepare results for CSV"""
    results = []
    true_factors = load_factors('factors.csv')
    for i,n in enumerate(numbers):
        print('#'*50)
        print(f"({i+1}/{len(numbers)})\nFactoring {n} ({n.nbits()} bits)... using {algorithm}")
        start = time.time()
        
        try:
            if algorithm == 'qs':
                factors = factor_qs(n)
            elif algorithm == 'mpqs':
                factors = mpqs(n, verbose=True)
            elif algorithm == 'rs':
                factors = factor_rs(n)
            else:
                raise ValueError(f"Invalid algorithm: {algorithm}")
        
            elapsed = time.time() - start
            print(f"Factors: {sorted(factors)}")
            print(f"True Factors: {true_factors.pop(1)}")
            print(f"Time: {elapsed:.2f}s")
            
            results.append({
                'number': str(n),
                'bits': n.nbits(),
                'time': elapsed,
                'factors': factors,
                'success': True,
                'valid': prod(factors) == n
            })
            
        except Exception as e:
            elapsed = time.time() - start
            print(f"Failed: {e}")
            results.append({
                'number': str(n),
                'bits': n.nbits(),
                'time': elapsed,
                'factors': 'Factorization failed',
                'success': False,
                'valid': False
            })
    return results

def save_results(results, filename):
    """Save to CSV"""
    with open(filename, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['number', 'bits', 'time', 'factors', 'success', 'valid'])
        writer.writeheader()
        writer.writerows(results)


def plot_runtimes(results, algorithm):
    """Plot runtime vs number size"""
    plt.figure(figsize=(10, 6))
    
    # Separate valid and unvalid factorizations
    valid = [r for r in results if r['valid']]
    failed = [r for r in results if not r['valid']]
    
    if valid:
        plt.scatter(
            [r['bits'] for r in valid],
            [r['time'] for r in valid],
            c='green',
            label=f'Valid ({len(valid)})'
        )
    
    if not valid:
        plt.scatter(
            [r['bits'] for r in failed],
            [r['time'] for r in failed],
            c='red',
            label=f'Not valid ({len(failed)})'
        )
    
    plt.xlabel('Number size (bits)')
    plt.ylabel('Runtime (seconds)')
    plt.title(f'{algorithm} Performance')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{algorithm.lower()}_runtime.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    # Load numbers from CSV
    numbers = load_numbers('verified_expanded_factorizations.csv')
    
    # Run random squares
    # results_rs = run_factorizations(numbers, algorithm='rs')
    # save_results(results_rs, filename='rs_results.csv')
    # plot_runtimes(results_rs, algorithm='rs')
    
    # # Run quadratic sieve
    # results_qs = run_factorizations(numbers, algorithm='qs')
    # save_results(results_qs, filename='qs_results.csv')
    # plot_runtimes(results_qs, algorithm='QS')

    #Run multiple quadratic sieve
    results_mpqs = run_factorizations(numbers, algorithm='mpqs')
    save_results(results_mpqs, filename='mpqs_results.csv')
    plot_runtimes(results_mpqs, algorithm='MPQS')
    
    print("\nResults saved to rs_results.csv, qs_results.csv and mpqs_results.csv")
