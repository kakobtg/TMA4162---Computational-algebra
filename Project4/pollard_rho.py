from math import gcd

def pollard_rho(n, max_iter=100000000):
    """
    Pollard's Rho algorithm for integer factorization.
    
    Parameters:
    n (int): The integer to factor.
    max_iter (int): Maximum number of iterations to prevent infinite loops.
    
    Returns:
    int: A non-trivial factor of n, or None if no factor is found.
    """
    if n % 2 == 0:
        return 2

    x = 2
    y = 2
    c = 1
    d = 1

    for _ in range(max_iter):
        x = (x * x + c) % n
        y = (y * y + c) % n
        y = (y * y + c) % n
        d = gcd(abs(x - y), n)
        
        if d > 1 and d < n:
            return d

    return None

if __name__ == "__main__":
    n = 1238926361552897 * 93461639715357977769163558199606896584051237541638188580280321 # F_8 Fermat number
    factor = pollard_rho(n)
    if factor:
        print(f"Found a non-trivial factor: {factor}")
    else:
        print("No factor found.")