from sage.all import *
from collections import defaultdict
import sys
sys.stdout.flush()


def is_prime(n):
    """primality test handling all integer types"""
    n = Integer(n)
    if n < 2: return False
    if n in (2, 3): return True
    if n % 2 == 0: return False
    return n.is_pseudoprime()

def safe_sqrt_mod(n, p):
    """Modular square root that always returns integers"""
    try:
        # Get all roots and force conversion to Python integers
        roots = Mod(n, p).sqrt(all=True)
        if isinstance(roots, (list, tuple)):
            return [int(r) for r in roots]
        return [int(roots)] if roots else []
    except (ValueError, TypeError):
        return []

def sieve_interval(n, M, factor_base):
    """ sieving """
    n = Integer(n)
    sqrt_n = isqrt(n)
    a = sqrt_n + 1
    sieve = [RDF(abs((a + i)**2 - n)).log() for i in range(-M, M + 1)]
    threshold = RDF(M * 20).log()
    
    for p in factor_base:
        p = Integer(p)
        if p == 2:
            # Special handling for p=2
            if n % 2 == 0:
                roots = [0]
            elif n % 8 == 1:
                roots = [0, 1]
            elif n % 8 == 5:
                roots = [0]
            else:
                continue
        else:
            roots = safe_sqrt_mod(n, p)
        
        for r in roots:
            # r is guaranteed to be a plain integer
            first_x = ((r - a) % p) + a
            
            # Adjust to be within sieving interval
            if first_x < a - M:
                first_x += p * ((a - M - first_x + p - 1) // p)
            start_idx = first_x - a + M
            
            if 0 <= start_idx < len(sieve):
                log_p = RDF(p).log()
                for idx in range(start_idx, len(sieve), p):
                    sieve[idx] -= log_p
    
    # Collect smooth relations
    smooth_relations = []
    for i, val in enumerate(sieve):
        if abs(val) < threshold:
            x = a + (i - M)
            q = x**2 - n
            if q == 0:
                return (x - sqrt_n, x + sqrt_n)
            smooth_relations.append((x, q))
    
    return smooth_relations

def quadratic_sieve(n, B_multiplier=2.0, max_attempts=5):
    """ QS implementation with safeguards"""
    n = Integer(n)
    if n < 2: raise ValueError("Number must be >1")
    if n % 2 == 0: return 2
    if is_prime(n): return n
    
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"\nAttempt {attempt}/{max_attempts}:")
            
            # Calculate parameters
            ln_n = RDF(n).log()
            ln_ln_n = ln_n.log()
            B = int(exp(0.6 * sqrt(ln_n * ln_ln_n)) * B_multiplier)
            B = max(B, 100)
            
            # Build factor base
            factor_base = [p for p in prime_range(B + 1) if p == 2 or kronecker(n, p) == 1]
            print(f"Using B = {B} with {len(factor_base)} primes")
            
            # Sieve with larger interval
            M = B * 50
            print(f"Sieving interval: ±{M}")
            relations = sieve_interval(n, M, factor_base)
            
            if isinstance(relations, tuple):
                return relations[0]
            
            print(f"Found {len(relations)} relations")
            if len(relations) < len(factor_base) + 10:
                raise ValueError("Insufficient relations")
            
            # Process relations
            factor_index = {p:i for i,p in enumerate(factor_base)}
            matrix = []
            processed = []
            
            for x, q in relations:
                factors = {}
                remaining = abs(q)
                for p in factor_base:
                    if remaining == 1: break
                    while remaining % p == 0:
                        factors[p] = factors.get(p, 0) + 1
                        remaining //= p
                if remaining == 1:
                    vec = [0]*len(factor_base)
                    for p, e in factors.items():
                        vec[factor_index[p]] = e % 2
                    matrix.append(vec)
                    processed.append((x, factors))
            
            # Linear algebra
            M = Matrix(GF(2), matrix).transpose()
            for basis in M.right_kernel().basis():
                indices = [i for i,x in enumerate(basis) if x]
                if len(indices) < 2: continue
                
                x = prod(processed[i][0] for i in indices) % n
                y_factors = defaultdict(int)
                for i in indices:
                    for p, e in processed[i][1].items():
                        y_factors[p] += e
                
                if all(e%2 == 0 for e in y_factors.values()):
                    y = prod(pow(p, e//2, n) for p,e in y_factors.items()) % n
                    for f in [gcd(x-y,n), gcd(x+y,n)]:
                        if 1 < f < n:
                            print(f"Found factor: {f}")
                            return f
            
            raise ValueError("No factors found")
            
        except ValueError as e:
            print(f"Attempt failed: {e}s)")
            B_multiplier *= 1.5
    
    raise ValueError(f"Failed after {max_attempts} attempts")



def factor_qs(n, _depth=0):
    """Complete factorization with proper return types"""
    n = Integer(n)
    indent = "  " * _depth
    
    # Base cases
    if n == 1:
        print(f"{indent}1 has no factors", flush=True)
        return []
    if is_prime(n):
        print(f"{indent}Found prime: {n}", flush=True)
        return [n]
    
    # Trial division
    for p in prime_range(2, 10**3 + 1):
        if n % p == 0:
            print(f"{indent}Found small factor: {p}", flush=True)
            return factor_qs(p, _depth+1) + factor_qs(n//p, _depth+1)
    
    # Quadratic Sieve
    try:
        print(f"{indent}Attempting QS on {n}...", flush=True)
        f = quadratic_sieve(n)
        if f:
            print(f"{indent}QS found factor: {f}", flush=True)
            return factor_qs(f, _depth+1) + factor_qs(n//f, _depth+1)
    except Exception as e:
        print(f"{indent}QS failed: {e}", flush=True)
    
    print(f"{indent}Could not factor: {n}", flush=True)
    return [n]  # Return as prime if all attempts fail
    