from sage.all import *
from itertools import product
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gram_schmidt(B):
    """Perform Gram-Schmidt orthogonalization."""
    return B.gram_schmidt()[0]

def babai_rounding_method(B, target):
    """Babai's rounding mathod for CVP"""
    G, _ = B.gram_schmidt()
    coeffs = B.solve_left(target)
    rounded_coeffs = vector(ZZ, [round(c) for c in coeffs])
    return B * rounded_coeffs

def enumerate_vectors(B, search_radius=10):
    """Enumerate all vectors in the lattice within a given search radius."""
    dim = B.ncols()
    min_length = float('inf')
    shortest_vector = None

    for coeffs in product(range(-search_radius, search_radius + 1), repeat=dim):
        if all(c == 0 for c in coeffs):
            continue
        candidate = B * vector(ZZ, coeffs)
        length = candidate.norm()
        if length < min_length:
            min_length = length
            shortest_vector = candidate

    return shortest_vector

def exhaustive_closest_vector_search(B, target, search_radius=50):
    """Exhaustively search for the closest vector to the target."""
    best_vector = None
    min_distance = float('inf')

    for coeffs in product(range(-search_radius, search_radius + 1), repeat=B.ncols()):
        candidate = B * vector(ZZ, coeffs)
        distance = (candidate - target).norm()
        if distance < min_distance:
            min_distance = distance
            best_vector = candidate

    return best_vector

def LLL_reduction(B):
    """Perform LLL reduction on the lattice basis from scratch."""
    delta = 0.75
    B_star, Mu = B.gram_schmidt()
    Mu = Mu.transpose()
    B_star = B_star.transpose()
    k = 2
    n = B.ncols()

    while k <= n:
        for j in range(k - 1, 0, -1):
            if abs(round(Mu[k][j])) > 0.5:
                B[k] -= q * B[j]
                Mu[k] -= q * Mu[j]
                B_star, Mu = B.gram_schmidt()
        if B[k].norm()**2 > (delta - Mu[k][k - 1]**2) * B_star[k - 1].norm()**2:
            k = k + 1
        else:
            B[k], B[k - 1] = B[k - 1], B[k]
            Mu[k], Mu[k - 1] = Mu[k - 1], Mu[k]
            k = max(k - 1, 2)
    return B

def compute_GSO(B):
    B_star, mu = B.gram_schmidt(orthonormal=False)
    d = [b.norm()**2 for b in B_star]  # Squared norms
    return d, mu


def RecursiveReduce(B, eta):
    """
    Input:
        B: A basis matrix (matrix over ZZ)
        eta: A reduction parameter in (0,1]
    Output:
        A reduced basis matrix
    """
    n = B.nrows()

    # Step 1: Find nr
    nr = n  # Assuming specific dimension always n for now, can adapt

    # Step 2: Compute GSO coefficients
    d, M = compute_GSO(B)

    # Step 3: Size-reduce B
    B = size_reduce(B, M)

    # Step 4: Define beta_bar
    max_norm = max(b.norm() for b in B.rows())
    beta_bar = 2 + ceil(log(sqrt(n) * max_norm**n, 2))

    # Step 5: Number of iterations
    num_iterations = (n//nr)**2 * ceil(log(beta_bar/eta)) if n != nr else 1

    for _ in range(num_iterations):

        for k in range(n - nr + 1):
            # Step 7: GSO block
            d_prime = d[k:k+nr]
            M_prime = [row[k:k+nr] for row in M[k:k+nr]]

            # Step 8: Recursive call
            U = RecursiveReduce_block(d_prime, M_prime, eta, beta_bar)

            # Step 9: Create U'
            U_prime = identity_matrix(ZZ, n)
            for i in range(nr):
                for j in range(nr):
                    U_prime[i+k, j+k] = U[i,j]

            # Step 10: Replace B by BU'
            B = B * U_prime

            # Update GSO
            d, M = B.gram_schmidt(orthonormal=False)

        # Step 11: Size-reduce again
        B = size_reduce(B, M)

    return B


def RecursiveReduce_block(d, M, eta, beta_bar):
    # This would be a reduced version operating on a small block.
    # Placeholder: just return identity matrix
    n = len(d)
    return identity_matrix(ZZ, n)


def size_reduce(B, M):
    n = B.nrows()
    for i in range(n):
        for j in range(i-1, -1, -1):
            mu = round(M[i][j])
            if mu != 0:
                B[i] -= mu * B[j]
    return B

    
            

def kannan_embedding(B, target, scale=1):
    """Construct Kannan's embedding lattice for solving CVP via SVP."""
    dim = B.ncols()
    augmented_B = B.augment(matrix(ZZ, dim, 1, [0] * dim))

    # Convert target vector to integers explicitly
    target_column = vector(ZZ, [round(x) for x in target] + [scale])

    B_prime = augmented_B.stack(target_column)
    return B_prime, scale

def decode_kannan_embedding(short_vector):
    """Decode the short vector from Kannan's embedding."""
    return short_vector[:-1]

def generate_lattice(dim): 
    """Generate a random lattice of given dimension."""
    return random_matrix(ZZ, dim, x=-10, y=10)



def plot_lattice(B):
    """Plot the lattice using matplotlib."""
    points = np.array(B.columns())
    plt.figure(figsize=(8, 8))
    plt.scatter(points[:, 0], points[:, 1], c='blue', marker='o')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid()
    plt.title('Lattice Points')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()
    
def main():
    # Example usage
    dim = 2
    B = generate_lattice(dim)
    target = vector(ZZ, [3, 4])
    print(f"Lattice B: {B}")
    print(f"Target vector: {target}")
    
    # Perform Gram-Schmidt orthogonalization
    G = gram_schmidt(B)
    
    # Babai's rounding method
    babai_vector = babai_rounding_method(B, target)
    
    # Enumerate vectors in the lattice
    enumerated_vector = enumerate_vectors(B)

    
    # Kannan's embedding
    B_prime, scale = kannan_embedding(B, target)
    
    # Decode the short vector from Kannan's embedding
    decoded_vector = decode_kannan_embedding(babai_vector)
    
    #recursive reduction
    eta = 0.75
    reduced_B2 = RecursiveReduce(B, eta)
    print(reduced_B2)
    # Perform exhaustive closest vector search
    closest_vector = exhaustive_closest_vector_search(B, target)
    print(f"Closest vector: {closest_vector}")
    
    # Plot the lattice
    plot_lattice(B)
    
if __name__ == "__main__":
    main()