from sage.all import *
from collections import defaultdict
import sys
import random
import time
sys.stdout.flush()

def is_prime(n):
    if n < 2: return False
    for p in [2,3,5,7,11,13,17,19,23,29,31,37]:
        if n % p == 0: return n == p
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in [2,325,9375,28178,450775,9780504,1795265022]: # Test deterministic integers for all 64 bits integers
        if a >= n: continue
        x = pow(a,d,n)
        if x == 1 or x == n-1: continue
        for _ in range(s-1):
            x = pow(x,2,n)
            if x == n-1: break
        else:
            return False
    return True

def factor_mpqs(n, _depth=0):
    n = Integer(n)
    indent = "  "*_depth
    
    if n == 1: return []
    if is_prime(n): return [n]
    
    
    for p in prime_range(2, 10**3 + 1):
        if n % p == 0:
            return sorted([p] + factor_mpqs(n//p, _depth+1))
    
    print(f"{indent}Attempting MPQS on {n}...")
    start_time = time.time()
    
    class MPQS:
        def __init__(self, n):
            self.n = n
            self.sqrt_n = isqrt(n)
            self.ln_n = RDF(n).log()
            self.ln_ln_n = self.ln_n.log()
            self.B = int(exp(0.8 * sqrt(self.ln_n * self.ln_ln_n)))  # Larger factor base
            self.M = self.B * 5  # Larger sieve interval
            self.factor_base = self._init_factor_base()
            self.smooth_relations = []
            self.threshold = None
            self.poly_queue = []
            self.current_poly = None
            self.a, self.b, self.c = None, None, None
            
        def _init_factor_base(self):
            fb = [p for p in prime_range(self.B + 1) if p == 2 or kronecker(self.n,p) == 1]
            print(f"{indent}Factor base: {len(fb)} primes (B={self.B})")
            return fb
        
        def _generate_poly(self):
            # Try to find a good 'a' value
            candidates = [p for p in self.factor_base if 10 < p < self.B//3]
            if not candidates:
                candidates = self.factor_base[5:50]
            
            a = 1
            a_factors = []
            # Use 3-5 primes for 'a'
            for p in random.sample(candidates, min(4, len(candidates))):
                a *= p
                a_factors.append(p)
            
            # Solve for b
            b = self._solve_b(a, a_factors)
            if b is None: 
                return False
            
            self.a = a
            self.b = b
            self.c = (b*b - self.n) // a
            return True
        
        def _generate_poly_batch(self, count=5):
            """Precompute several polynomials at once."""
            self.poly_queue = []
            for _ in range(count):
                if self._generate_poly():
                    self.poly_queue.append((self.a, self.b, self.c))

        def _switch_poly(self):
            """Switch to the next precomputed polynomial."""
            if not self.poly_queue:
                self._generate_poly_batch()
            self.a, self.b, self.c = self.poly_queue.pop()
            print(f'{indent}Switched to polynomial: a={self.a}, b={self.b}, c={self.c}')
        
        def _solve_b(self, a, factors):
            congruences = []
            for p in factors:
                if p == 2:
                    roots = [1] if self.n % 8 == 1 else []
                else:
                    roots = Mod(self.n,p).sqrt(all=True)
                if not roots: return None
                congruences.append((int(roots[0]), p))
            return crt([r for r,p in congruences], [p for r,p in congruences]) % a
        
        def _sieve(self):
            a, b, c = self.a, self.b, self.c
            sieve = [0.0]*(2*self.M + 1)
            log_p = {p:log(float(p)) for p in self.factor_base}
            self.threshold = log(float(self.M * 20))
            
            # Initialize sieve array
            for i in range(-self.M, self.M + 1):
                val = abs(a*i*i + 2*b*i + c)
                sieve[i + self.M] = log(float(val)) if val != 0 else 0.0
            
            # Sieve with each prime
            for p in self.factor_base:
                if p == 2:  # Special case
                    if self.n % 2 == 0:
                        roots = [0]
                    elif self.n % 8 == 1:
                        roots = [0,1]
                    elif self.n % 8 == 5:
                        roots = [0]
                    else:
                        continue
                else:
                    roots = Mod(self.n,p).sqrt(all=True)
                
                for root in roots:
                    r = int(root)
                    if gcd(a, p) != 1:
                        continue
                    inv_a = inverse_mod(a, p)
                    x0 = (-b + r) * inv_a % p
                    # Sieve progression
                    x = x0
                    while x < -self.M: x += p
                    while x > self.M: x -= p
                    start = x if x >= -self.M else x + p
                    for idx in range(start + self.M, 2*self.M + 1, p):
                        sieve[idx] -= log_p[p]
            
            # Check for smooth numbers
            smooth = []
            for i in range(2*self.M + 1):
                if sieve[i] < self.threshold:
                    x = i - self.M
                    q = a*x*x + 2*b*x + c
                    if q == 0: return (x - self.sqrt_n, x + self.sqrt_n)
                    factors = self._factor(abs(q))
                    if factors: 
                        self.smooth_relations.append((x, factors))
                        if len(self.smooth_relations) % 10 == 0:
                            print(f"\r{indent}Found {len(self.smooth_relations)} relations", end='')
            return len(self.smooth_relations)
        
        def _factor(self, m):
            factors = {}
            for p in self.factor_base:
                while m % p == 0:
                    factors[p] = factors.get(p,0) + 1
                    m = m // p
            return factors if m == 1 else None
        
        def solve(self):
            required_relations = len(self.factor_base) + 20
            
            # Precompute initial batch of polynomials
            self._generate_poly_batch()
            self._switch_poly()  # Must call this before first sieve!
            
            while len(self.smooth_relations) < required_relations:
                new_rels = self._sieve()
                
                if isinstance(new_rels, tuple):  # Factor found
                    return new_rels[0]
                
                if new_rels == 0:  # No new relations
                    if self.poly_queue:
                        self._switch_poly()
                    else:
                        self.M = int(self.M * 2)  # Double sieve range
                        print(f"{indent}Increasing M to {self.M}")
                        self._generate_poly_batch()
                        self._switch_poly()
            
            print(f"\n{indent}Processing {len(self.smooth_relations)} relations...")
            return self._process_relations()
        
        def _process_relations(self):
            matrix = []
            fb_index = {p:i for i,p in enumerate(self.factor_base)}
            
            for x, factors in self.smooth_relations:
                row = [0]*len(self.factor_base)
                for p, exp in factors.items():
                    row[fb_index[p]] = exp % 2
                matrix.append(row)
            
            M = Matrix(GF(2), matrix).transpose()
            for vec in M.right_kernel().basis():
                x_prod = 1
                y_factors = defaultdict(int)
                for i, val in enumerate(vec):
                    if val:
                        x, factors = self.smooth_relations[i]
                        x_prod = (x_prod * (self.a*x + self.b)) % self.n
                        for p, exp in factors.items():
                            y_factors[p] += exp
                
                y = 1
                for p, exp in y_factors.items():
                    if exp % 2 != 0: break
                    y = (y * pow(p, exp//2, self.n)) % self.n
                else:
                    f = gcd(x_prod - y, self.n)
                    if 1 < f < self.n:
                        print(f"{indent}Found factor: {f} (in {time.time()-start_time:.1f}s)")
                        return f
            return None
    
    mpqs = MPQS(n)
    f = mpqs.solve()
    if f:
        return sorted(factor_mpqs(f, _depth+1) + factor_mpqs(n//f, _depth+1))
    
    print(f"{indent}MPQS failed after {time.time()-start_time:.1f} seconds")
    return [n]

# testing
n = 787932673709 * 987932681039
print(f"Factoring {n}...")
factors = factor_mpqs(n)
print(f"Final factors: {factors}")
