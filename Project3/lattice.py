import time
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import pandas as pd
from IPython.display import display



def gram_schmidt(B):
    """
    Computes the Gram-Schmidt orthogonalization of B.
    Returns B_star (orthogonal basis) and mu (Gram-Schmidt coefficients).
    """
    dim = B.shape[1]
    B_star = np.zeros_like(B, dtype=np.float64)
    mu = np.zeros((dim, dim), dtype=np.float64)
    norms = np.zeros(dim, dtype=np.float64)

    for i in range(dim):
        B_star[:, i] = B[:, i]
        for j in range(i):
            mu[i, j] = np.dot(B[:, i], B_star[:, j]) / np.dot(B_star[:, j], B_star[:, j]) + 1e-6
            B_star[:, i] -= mu[i, j] * B_star[:, j]
        
        norms[i] = np.dot(B_star[:, i], B_star[:, i])

    return B_star, mu, norms


def babai_rounding_method(B, target):
    """
    Babai's Rounding Method to find the closest lattice point to the target vector.
    """
    Q, R = np.linalg.qr(B) # Q = othogonal, R = upper triangular
    coefficients = np.linalg.solve(R, Q.T @ target)
    rounded_coeffs = np.round(coefficients)
    closest_vector = B @ rounded_coeffs
    return closest_vector

def exhaustive_shortest_vector_search(B):
    """
    Finds the shortest nonzero vector in the lattice spanned by B using an exhaustive search.

    Parameters:
        B: np.ndarray - The lattice basis matrix.
        search_radius: int - The range of coefficients to search.

    Returns:
        The shortest nonzero vector found in the lattice.
    """
    dim = B.shape[1]
    min_length = float('inf')
    shortest_vector = None
    estimated_radius = int(np.linalg.norm(B, axis=1).mean() / 2)
    search_radius = max(2, min(4, estimated_radius))


    for coeffs in product(range(-search_radius, search_radius + 1), repeat=dim):
        coeffs = np.array(coeffs)
        if np.all(coeffs == 0):  # Skip the zero vector
            continue
        
        candidate = B @ coeffs
        length = np.linalg.norm(candidate)
        
        if length < min_length:
            min_length = length
            shortest_vector = candidate
        if min_length < 1e-6:
            break

    return shortest_vector


def shortest_vector_search(B, max_attempts=1000):
    """
    Finds an approximation of the shortest vector using a greedy method with limited iterations.
    """
    B = lll_reduction(B)  # Reduce first
    shortest = B[:, 0]  # Start with the first basis vector
    min_length = np.linalg.norm(shortest)
    
    attempts = 0
    for i in range(1, B.shape[1]):
        candidate = B[:, i]
        length = np.linalg.norm(candidate)
        if length < min_length:
            shortest = candidate
            min_length = length

        attempts += 1
        if attempts > max_attempts:
            break

    return shortest



def exhaustive_closest_vector_search_2(B, target, search_radius=50):
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

def exhaustive_closest_vector_search(B, target, search_radius=3):
    """
    Faster exhaustive closest vector search using a sorted, prioritized search order.
    """
    dim = B.shape[1]
    best_vector = None
    min_distance = float('inf')

    # Sort basis vectors by norm (shorter vectors first, for better early solutions)
    norms = np.linalg.norm(B, axis=0)
    order = np.argsort(norms)  # Prioritize shorter basis vectors
    B = B[:, order]  

    def dfs(coeffs, index):
        nonlocal best_vector, min_distance

        if index == dim:
            candidate = B @ np.array(coeffs)
            distance = np.linalg.norm(candidate - target)
            if distance < min_distance:
                min_distance = distance
                best_vector = candidate
            return
        
        for c in range(-search_radius, search_radius + 1):
            new_coeffs = coeffs + [c]
            
            # Estimate early if this is worth continuing
            partial_vector = B[:, :index+1] @ np.array(new_coeffs)
            partial_target = target[:partial_vector.shape[0]]  # Ensure consistent shape
            partial_distance = np.linalg.norm(partial_vector - partial_target)

            if partial_distance > min_distance:
                continue  # Skip bad branches
            
            dfs(new_coeffs, index + 1)

    dfs([], 0)
    return best_vector



def kannan_embedding(B, target):
    """Construct Kannan's embedding lattice for solving CVP via SVP."""
    scale = np.linalg.norm(B) / np.linalg.norm(target)
    dim = B.shape[1]
    augmented_B = np.hstack((B, np.zeros((dim, 1))))
    target_column = np.append(target, scale).reshape(1, -1)  # Fix reshaping issue
    return np.vstack((augmented_B, target_column)), scale


def decode_kannan_embedding(B_prime, short_vector):
    """Decode the closest vector from Kannan's embedding method."""
    return short_vector[:-1]


def lll_reduction(B, delta=0.75, max_swaps=5000):
    """
    LLL lattice basis reduction
    """
    B = B.astype(np.float64)
    dim = B.shape[1]
    
    # Compute Gram-Schmidt    
    B_star, mu, norms = gram_schmidt(B)
    
    # LLL Reduction
    k = 1
    swap_count = 0
    while k < dim and swap_count < max_swaps:
        # Size reduction
        for j in range(k - 1, -1, -1):
            if np.isnan(mu[k, j]):  # If mu is NaN, stop early
                print(f"NaN detected in LLL reduction at k={k}, j={j}. Stopping.")
                return B
            
            q = np.float64(round(mu[k, j]))
            if q != 0:
                B[:, k] -= (q * B[:, j])
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
            swap_count += 1
               
    return B


def bkz_reduction(B, delta=0.75, max_iterations=20):
    """
    Improved BKZ (Block Korkine-Zolotarev) lattice basis reduction with better stability.
    """
    dim = B.shape[1]
    B = B.astype(np.float64)
    
    block_size = max(3, min(dim // 4, 10))  # Dynamic block size but not too small

    print(f"Using block size {block_size} for dimension {dim} in BKZ")
    
    B /= np.linalg.norm(B, axis=0, keepdims=True)
    
    for _ in range(max_iterations):
        for i in range(0, dim - block_size + 1):
            block = B[:, i:i+block_size].copy()  # Extract block

            # Catch SVD convergence failure
            try:
                if np.log10(np.linalg.cond(block)) > 8:  # Instead of checking > 1e8
                    #print(f"Skipping unstable block at i={i}")
                    continue
            except np.linalg.LinAlgError:
                #print(f"Skipping block at i={i} due to SVD non-convergence.")
                continue

            # Apply LLL inside the block instead of QR
            reduced_block = lll_reduction(block, delta)
            B[:, i:i+block_size] = reduced_block  # Replace reduced block

        # Apply global LLL after each iteration
        B = lll_reduction(B, delta)
    
    return B




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

    # Create an explicit unimodular matrix
    unimodular = np.eye(dim, dtype=int)
    for i in range(dim):
        for j in range(i):
            unimodular[i, j] = np.random.randint(-2, 3)  # Small integer values

    return B @ unimodular

            
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
    
    return lattice_point + offset, lattice_point

def save_results_to_file(results, filename="lattice_results.csv"):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def run_experiment(dimensions, time_limit=12*3600):
    """
    Runs an experiment on CVP and SVP across increasing dimensions.

    Parameters:
        dimensions (list): List of lattice dimensions to test.
        time_limit (int): Maximum runtime in seconds.
    """

    results = []
    babai_times, kannan_times, shortest_times, lll_times, bkz_times = [], [], [], [], []
    total_time = 0

    for dim in dimensions:
        print(f"\n--- Running experiment for dimension {dim} ---")
        start_exp = time.time()

        # Generate lattice and target vector
        B = generate_bad_lattice(dim)
        v, correct_closest_vector = generate_bad_target_vector(B)

        ### Step 1: Compute Closest Vector using Babai’s rounding (Original Basis)
        start = time.time()
        babai_original = exhaustive_closest_vector_search(B, v)
        babai_time = time.time() - start
        babai_times.append(babai_time)

        ### Step 2: Reduce Basis using LLL
        start = time.time()
        reduced_B_LLL = lll_reduction(B, delta=0.75)
        lll_time = time.time() - start
        lll_times.append(lll_time)

        ### Step 2.5: BKZ reduction
        '''
        start = time.time()
        reduced_B_BKZ = bkz_reduction(B)
        bkz_time = time.time() - start
        bkz_times.append(bkz_time)
        '''

        ### Step 3: Compute Closest Vector using Babai’s rounding (Reduced Basis)
        start = time.time()
        if dim <= 8:
            closest_reduced_LLL = exhaustive_closest_vector_search(reduced_B_LLL, v)
            #closest_reduced_BKZ = exhaustive_closest_vector_search(reduced_B_BKZ, v)
        else:
            closest_reduced_LLL = babai_rounding_method(reduced_B_LLL, v)
            #closest_reduced_BKZ = babai_rounding_method(reduced_B_BKZ, v)
        closest_reduced_time = time.time() - start
        babai_times.append(closest_reduced_time)

        ### Step 4: Compute Closest Vector using Kannan’s embedding
        start = time.time()
        B_prime_LLL, scale = kannan_embedding(reduced_B_LLL, v)
        kannan_short_vector_LLL = babai_rounding_method(B_prime_LLL, np.append(v, scale))
        kannan_closest_decoded_LLL = decode_kannan_embedding(B_prime_LLL, kannan_short_vector_LLL)

        #B_prime_BKZ, scale = kannan_embedding(reduced_B_BKZ, v)
        #kannan_short_vector_BKZ = babai_rounding_method(B_prime_BKZ, np.append(v, scale))
        #kannan_closest_decoded_BKZ = decode_kannan_embedding(B_prime_BKZ, kannan_short_vector_BKZ)
        
        kannan_time = time.time() - start
        kannan_times.append(kannan_time)

        ### Step 5: Compute Shortest Lattice point
        start = time.time()
        if dim <= 8:
            shortest_original = exhaustive_shortest_vector_search(B)
            shortest_reduced_LLL = exhaustive_shortest_vector_search(reduced_B_LLL)
            #shortest_reduced_BKZ = exhaustive_shortest_vector_search(reduced_B_BKZ)
        else:
            shortest_original = shortest_vector_search(B)
            shortest_reduced_LLL = shortest_vector_search(reduced_B_LLL)
            #shortest_reduced_BKZ = shortest_vector_search(reduced_B_BKZ)
        shortest_time = time.time() - start
        shortest_times.append(shortest_time)

        ### Step 6: Compare Results
        kannan_correct_LLL = np.allclose(correct_closest_vector, kannan_closest_decoded_LLL, atol=1e-5)
        #kannan_correct_BKZ = np.allclose(correct_closest_vector, kannan_closest_decoded_BKZ, atol=1e-3)
        
        
        babai_correct_norm = np.linalg.norm(correct_closest_vector - babai_original)
        #kannan_correct_norm_BKZ = np.linalg.norm(correct_closest_vector - kannan_closest_decoded_BKZ)
        kannan_correct_norm_LLL = np.linalg.norm(correct_closest_vector - kannan_closest_decoded_LLL)

        print(f"Kannan LLL correct: {kannan_correct_LLL}")
        #print(f"Kannan BKZ correct: {kannan_correct_BKZ}")
        print(f"Babai norm: {babai_correct_norm}")
        print(f"Kannan LLL norm: {kannan_correct_norm_LLL}")
        #print(f"Kannan BKZ norm: {kannan_correct_norm_BKZ}")


        babai_improvement_LLL = np.linalg.norm(babai_original - v) - np.linalg.norm(closest_reduced_LLL - v)
        #babai_improvement_BKZ = np.linalg.norm(babai_original - v) - np.linalg.norm(closest_reduced_BKZ - v)

        shortest_improvement_LLL = np.linalg.norm(shortest_original) - np.linalg.norm(shortest_reduced_LLL)
        #shortest_improvement_BKZ = np.linalg.norm(shortest_original) - np.linalg.norm(shortest_reduced_BKZ)
    

        results.append({
                "Dim": dim,
                "Babai Error (Original)": np.linalg.norm(babai_original - correct_closest_vector),
                "Babai Error (LLL)": np.linalg.norm(closest_reduced_LLL - correct_closest_vector),
                #"Babai Error (BKZ)": np.linalg.norm(closest_reduced_BKZ - correct_closest_vector),
                "Kannan Error (LLL)": np.linalg.norm(kannan_closest_decoded_LLL - correct_closest_vector),
                #"Kannan Error (BKZ)": np.linalg.norm(kannan_closest_decoded_BKZ - correct_closest_vector),
                "Babai Improvement (LLL)": np.linalg.norm(babai_original - correct_closest_vector) - np.linalg.norm(closest_reduced_LLL - correct_closest_vector),
                #"Babai Improvement (BKZ)": np.linalg.norm(babai_original - correct_closest_vector) - np.linalg.norm(closest_reduced_BKZ - correct_closest_vector),
                "Shortest Norm (Original)": np.linalg.norm(shortest_original),
                "Shortest Norm (LLL)": np.linalg.norm(shortest_reduced_LLL),
                #"Shortest Norm (BKZ)": np.linalg.norm(shortest_reduced_BKZ),
                "Babai Time": babai_time,
                "LLL Time": lll_time,
                #"BKZ Time": bkz_time,
                "Kannan Time": kannan_time,
                "Shortest Time": shortest_time,
                "Babai Correct": np.allclose(babai_original, correct_closest_vector, atol=1e-5),
                "Kannan LLL Correct": np.allclose(kannan_closest_decoded_LLL, correct_closest_vector, atol=1e-5),
                #"Kannan BKZ Correct": np.allclose(kannan_closest_decoded_BKZ, correct_closest_vector, atol=1e-5)
            })


        print(f"Dimension {dim} completed in {time.time() - start_exp:.2f} seconds.")

        total_time += time.time() - start_exp
        if total_time > time_limit:
            print(f"Time limit reached. Stopping at dimension {dim}.")
            break

    save_results_to_file(results)


    plt.figure(figsize=(8, 6))
    #plt.plot(dimensions[:len(results)], [r["BKZ Time"] for r in results], label='BKZ Reduction', marker='s')
    plt.plot(dimensions[:len(results)], [r["LLL Time"] for r in results], label='LLL Reduction', marker='x')
    plt.yscale('log')
    plt.xlabel('Lattice Dimension')
    plt.ylabel('Runtime (seconds)')
    plt.legend()
    plt.title('Runtime Comparison of LLL vs BKZ')
    plt.show()

run_experiment(dimensions=[2,4,6,8,10,20,25,32,50,64, 100])