from factorization.quadratic_sieve_sage import quadratic_sieve
#from factorization.random_squares import random_squares_method

import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
import re
from sage.all import Integer, is_prime, prod

def parse_number(num_str):
    """Safely evaluate numbers like 2**79 + 2**40 + 1"""
    try:
        return Integer(eval(num_str.replace(' ', '')))
    except:
        return None

def load_and_analyze(filename, use_qs=False):
    """Load CSV and analyze factorizations"""
    df = pd.read_csv(filename)
    df['n'] = df['Number'].apply(parse_number)
    df = df.dropna()  # Remove rows with invalid numbers
    
    # Verify reported factors
    df['reported_factors'] = df['Factors'].apply(literal_eval)
    df['verified'] = df.apply(
        lambda row: prod(row['reported_factors']) == row['n'] and 
                   all(is_prime(f) for f in row['reported_factors']), 
        axis=1
    )
    
    if use_qs:
        # Use QS to correct wrong factorizations
        df['correct_factors'] = None
        df['factor_time'] = 0.0
        
        for idx in df[~df['verified']].index:
            n = df.at[idx, 'n']
            factors, time_taken = factor_using_qs(n)
            df.at[idx, 'correct_factors'] = str(factors)
            df.at[idx, 'factor_time'] = time_taken
    
    return df

def plot_results(df):
    """Generate analysis plots"""
    df['bits'] = df['n'].apply(lambda x: x.nbits())
    
    plt.figure(figsize=(12,5))
    
    # Accuracy plot
    plt.subplot(1,2,1)
    df['verified'].value_counts().plot(kind='bar')
    plt.xticks([0,1], ['Incorrect', 'Correct'], rotation=0)
    plt.title('Factorization Accuracy')
    
    # Runtime plot (if QS was used)
    if 'factor_time' in df:
        plt.subplot(1,2,2)
        plt.scatter(df['bits'], df['factor_time'], alpha=0.7)
        plt.xlabel('Number size (bits)')
        plt.ylabel('Factorization time (s)')
        plt.title('QS Runtime vs Number Size')
    
    plt.tight_layout()
    plt.savefig('factorization_analysis.png')
    plt.show()

if __name__ == "__main__":
    df = load_and_analyze('factors.csv', use_qs=True)
    print(f"Verification results: {sum(df['verified'])}/{len(df)} correct")
    plot_results(df)
    df.to_csv('verified_results.csv', index=False)