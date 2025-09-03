import time
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import pandas as pd


def gram_schmidt(B): # function for the reorthogonalization
    Q, R = np.linalg.qr(B)
    return Q @ R


def babai_rounding_method(B, target):
    """
    Babai's Rounding Method to find the closest lattice point to the target vector.
    """
    Q, R = np.linalg.qr(B) # Q = othogonal, R = upper triangular
    coefficients = np.linalg.solve(R, Q.T @ target)
    rounded_coeffs = np.round(coefficients)
    closest_vector = B @ rounded_coeffs
    return closest_vector

def shortest_vector_search(B, search_radius=10):
    """
    Find the shortest nonzero vector in the lattice spanned by B.

    Parameters:
    - B: np.ndarray, the lattice basis matrix (dim x dim)
    - search_radius: int, the range of coefficients to search (larger means more exhaustive)

    Returns:
    - shortest_vector: np.ndarray, the shortest nonzero vector found
    """
    dim = B.shape[1]
    min_length = float('inf')
    shortest_vector = None

    for coeffs in product(range(-search_radius, search_radius + 1), repeat=dim):
        if all(c == 0 for c in coeffs):  # Skip the zero vector
            continue
        
        candidate = B @ np.array(coeffs)
        length = np.linalg.norm(candidate)
        
        if length < min_length:
            min_length = length
            shortest_vector = candidate

    return shortest_vector


def exhaustive_closest_vector_search(B, target, search_radius=50):
    """
    Exhaustive search for the closest lattice point to the target vector.
    """
    best_vector = None
    min_distance = float('inf')
    for coeffs in product(range(-search_radius, search_radius + 1), repeat=B.shape[1]):
        candidate = B @ np.array(coeffs)
        distance = np.linalg.norm(candidate - target)
        if distance < min_distance:
            min_distance = distance
            best_vector = candidate
    return best_vector


def kannan_embedding(B, target, scale=1):
    """Construct Kannan's embedding lattice for solving CVP via SVP."""
    dim = B.shape[1]
    augmented_B = np.hstack((B, np.zeros((dim, 1))))
    target_column = np.append(target, scale).reshape(1, -1)  # Fix reshaping issue
    return np.vstack((augmented_B, target_column)), scale


def decode_kannan_embedding(B_prime, short_vector):
    """Decode the closest vector from Kannan's embedding method."""
    return short_vector[:-1]

def lll_reduction(B, delta=0.75):
    """
    Fast LLL lattice basis reduction using only basic linear algebra.
    """
    B = B.astype(np.float64)
    dim = B.shape[1]
    
    # Compute Gram-Schmidt
    B_star = np.zeros_like(B)
    mu = np.zeros((dim, dim))
    norms = np.zeros(dim)
    
    B_star[:, 0] = B[:, 0]
    norms[0] = np.dot(B_star[:, 0], B_star[:, 0])
    
    for i in range(1, dim):
        B_star[:, i] = B[:, i]
        for j in range(i):
            mu[i, j] = np.dot(B[:, i], B_star[:, j]) / norms[j]
            B_star[:, i] -= mu[i, j] * B_star[:, j]
        norms[i] = np.dot(B_star[:, i], B_star[:, i])
    
    # LLL Reduction
    k = 1
    while k < dim:
        # Size reduction
        for j in range(k - 1, -1, -1):
            q = round(mu[k, j])
            if q != 0:
                B[:, k] -= q * B[:, j]
                mu[k, j] -= q
        
        # Lovász condition check
        if norms[k] >= (delta - mu[k, k - 1]**2) * norms[k - 1]:
            k += 1
        else:
            # Swap B_k and B_{k-1}
            B[:, [k, k - 1]] = B[:, [k - 1, k]]
            B_star[:, [k, k - 1]] = B_star[:, [k - 1, k]]
            mu[k, k - 1], mu[k - 1, k - 1] = mu[k - 1, k - 1], mu[k, k - 1]
            
            # Recompute norms
            norms[k], norms[k - 1] = norms[k - 1], np.dot(B_star[:, k - 1], B_star[:, k - 1])
            
            k = max(1, k - 1)
    
    return B



def lll_reduction2(B, delta=0.75):
    """
    Lenstra-Lenstra-Lovász (LLL) lattice basis reduction algorithm.
    """
    B = B.copy().astype(np.float64)
    n = B.shape[0]
    m = B.shape[1]
    #B /= np.linalg.norm(B, axis=0)  # Normalize each column
    Q, R = np.linalg.qr(B)
    mu = np.zeros((m, m))
    for i in range(m):
        for j in range(i):
            mu[i, j] = R[j, i] / R[j, j]
    
    
    k = 1
    swap_count = 0
    
    while k < m:
        for j in range(k - 1, -1, -1):
            q = round(mu[k, j])
            if q != 0:
                B[:, k] -= q * B[:, j]
                for i in range(j + 1): # only need to do this up to col j
                    mu[k, i] -= q * mu[j, i]
                R[j,k] -= q * R[j,j] # also update the QR decomp, which is much more efficient
                for i in range(j):
                  R[i,k] -= q*R[i,j]
        
        # 3. Lovász Condition
        # Use the Gram-Schmidt coefficients directly
        if R[k,k]**2 >= (delta - mu[k, k-1]**2) * R[k-1,k-1]**2: #R[k,k]**2: # np.dot(Q[:,k], B[:,k])
            k += 1
        else:
            # Swap columns in B
            B[:, [k, k - 1]] = B[:, [k - 1, k]]
            # recompute just the column entries that we need to update
            R = np.linalg.qr(B, mode='r')

            # Update mu accordingly
            for i in range(m):
                for j in range(i):
                    mu[i, j] = R[j, i] / R[j, j]

            k = max(1, k - 1)
            swap_count += 1
            
        if swap_count % 5 == 0 and swap_count > 0: # reorthogonalization
            B = gram_schmidt(B)
            Q, R = np.linalg.qr(B) # have to recalculate the QR
            for i in range(m):
                for j in range(i):
                    mu[i, j] = R[j, i] / R[j, j] # have to recalculate mu
    print(f'Swaps in dim {n}: {swap_count}')
    return B


def KZ_reduction(B):
    B_star = np.zeros_like()
    n = B.shape[1]
    Q, R = np.linalg.qr(B)
    
    mu = np.zeros((n,n))
    
    for i in range(n):
        for j in range(i):
            mu[i,j] = np.dot(B[:, i], Q[:, j]) / np.dot(Q[:, j], Q[:, j])
    pass
    
    

def generate_lattice(dim):
    """Generate a random integral lattice basis of given dimension."""
    return np.random.randint(-10, 10, size=(dim, dim), dtype=np.int64)

def generate_bad_lattice(dim):
    """
    Generates a poorly conditioned lattice basis.
    """
    while True:
        # Generate a lower triangular matrix with integer entries
        B = np.tril(np.random.randint(-10, 10, size=(dim, dim)))

        # Ensure full rank
        if np.linalg.matrix_rank(B) == dim:
            break

    # Create a unimodular matrix (integer determinant ±1)
    unimodular = np.eye(dim, dtype=int) + np.random.randint(-2, 3, size=(dim, dim))
    
    # Ensure determinant is ±1
    while np.linalg.det(unimodular) != 1 and np.linalg.det(unimodular) != -1:
        unimodular = np.eye(dim, dtype=int) + np.random.randint(-2, 3, size=(dim, dim))

    return B @ unimodular


def generate_better_lattice(dim, max_entry=10):
    """
    Generates a lattice basis with a better condition number.
    """
    while True:
        B = np.random.randint(-max_entry, max_entry + 1, size=(dim, dim))
        if np.linalg.matrix_rank(B) == dim:  # Ensure full rank
            # Check condition number (optional, but good practice)
            cond_num = np.linalg.cond(B)
            if cond_num < 1e6:  # Adjust threshold as needed
                return B
            
def generate_bad_target_vector(B, scale_factor=0.1):
    """
    Generates a target vector near the lattice but not in it.
    """
    dim = B.shape[1]
    
    # Scale coefficients based on column norms to keep target close
    column_norms = np.linalg.norm(B, axis=0)
    coeffs = np.random.uniform(-2, 2, size=dim) * column_norms
    
    lattice_point = B @ np.round(coeffs)  # Closest lattice point
    
    # Small random offset in a random direction
    offset = np.random.uniform(-scale_factor, scale_factor, size=dim) * column_norms
    
    return lattice_point + offset

def run_experiment():
    """Run the experiment for increasing lattice dimensions."""
    dimensions = [2, 4, 6, 8, 16, 20, 25, 32]
    results = []
    babai_times, kannan_times, shortest_times, exhaustive_times = [], [], [], []

    for dim in dimensions:
        print(f"\n--- Running experiment for dimension {dim} ---")
        B = generate_bad_lattice(dim)
        v = generate_bad_target_vector(B)

        # (i) Closest Vector using Babai (Original Basis)
        start = time.time()
        babai_original = babai_rounding_method(B, v)
        babai_times.append(time.time() - start)

        # Exhaustive Search (only for small dimensions)
        if dim <= 8:
            start = time.time()
            exhaustive_closest = exhaustive_closest_vector_search(B, v, search_radius=3)
            exhaustive_times.append(time.time() - start)
        else:
            exhaustive_closest = None
            exhaustive_times.append(None)

        # (ii) LLL Reduction
        reduced_B = lll_reduction(B, delta=0.75)

        # (iii) Closest Vector using Kannan (Reduced Basis)
        start = time.time()
        B_prime, scale = kannan_embedding(reduced_B, v)
        kannan_short_vector = babai_rounding_method(B_prime, np.append(v, scale))
        kannan_closest_decoded = decode_kannan_embedding(B_prime, kannan_short_vector)
        kannan_times.append(time.time() - start)

        # (iv) Shortest Vector Search
        start = time.time()
        shortest_original = shortest_vector_search(B, search_radius=3)
        shortest_reduced = shortest_vector_search(reduced_B, search_radius=3)
        shortest_times.append(time.time() - start)

        # Comparisons
        babai_kannan_match = np.allclose(babai_original, kannan_closest_decoded, atol=1e-5)
        if exhaustive_closest is not None:
            babai_correct = np.allclose(babai_original, exhaustive_closest, atol=1e-5)
            kannan_correct = np.allclose(kannan_closest_decoded, exhaustive_closest, atol=1e-5)
        else:
            babai_correct = kannan_correct = None

        # Results for each dimension
        results.append({
            "Dim": dim,
            "Babai Error (Original)": np.linalg.norm(babai_original - v),
            #"Babai Error (Reduced)": np.linalg.norm(babai_reduced - v),
            "Kannan Error": np.linalg.norm(kannan_closest_decoded - v),
            "Babai-Kannan Match": babai_kannan_match,
            "Shortest Norm (Original)": np.linalg.norm(shortest_original),
            "Shortest Norm (Reduced)": np.linalg.norm(shortest_reduced),
            "Babai Correct?": babai_correct,
            "Kannan Correct?": kannan_correct
        })

        print(f"Dimension {dim} completed.\n")

    # Display results
    df = pd.DataFrame(results)
    print("\nFinal Results:")
    print(df.to_string(index=False))

    # Convert results to DataFrame and display
    df = pd.DataFrame(results)
    print("\nFinal Results:")
    print(df.to_string(index=False))

    
    
    plt.plot(dimensions, babai_times, label='Babai Rounding Method', marker='o')
    plt.plot([d for d in dimensions if d <= 8], [t for t in exhaustive_times if t is not None], label='Exhaustive Search', marker='x')
    plt.plot(dimensions, kannan_times, label='Kannan Embedding Method', marker='s')
    plt.plot([d for d in dimensions if d <= 8], [t for t in shortest_times if t is not None], label='Shortest Vector Search', marker='d')
    plt.yscale('log')
    plt.xlabel('Lattice Dimension')
    plt.ylabel('Runtime (seconds)')
    plt.legend()
    plt.title('Runtime Comparison of Closest and Shortest Vector Algorithms')
    plt.show()
    

run_experiment()
