import unittest
import numpy as np
from itertools import product  # Import if not already imported
from lattice2 import *

class LatticeTests(unittest.TestCase):

    def test_known_closest_vector(self):
        """Test cases where the closest vector is known a priori."""

        # Test Case 1: Target vector *is* a lattice point.
        B1 = np.array([[2, 0], [1, 3]])
        v1 = np.array([4, 6])  # v1 = 2 * B1[:, 0] + 2 * B1[:, 1]
        closest1_babai = babai_rounding_method(B1, v1)
        closest1_kannan = decode_kannan_embedding(*kannan_embedding(B1, v1))
        self.assertTrue(np.allclose(closest1_babai, v1))
        self.assertTrue(np.allclose(closest1_kannan, v1))

        # Test Case 2:  Target vector is close to a known lattice point.
        B2 = np.array([[1, 0], [0, 1]])  # Simple integer lattice
        v2 = np.array([2.1, 3.8])
        expected_closest2 = np.array([2, 4])  # Known closest point
        closest2_babai = babai_rounding_method(B2, v2)
        closest2_kannan = decode_kannan_embedding(*kannan_embedding(B2, v2))
        self.assertTrue(np.allclose(closest2_babai, expected_closest2))
        self.assertTrue(np.allclose(closest2_kannan, expected_closest2))

        # Test Case 3: Higher dimension, known coefficients.
        B3 = generate_better_lattice(5)  # Use your lattice generator
        coeffs3 = np.array([1, -2, 0, 3, -1])  # Known integer coefficients
        v3 = B3 @ coeffs3  # Target vector is in the lattice
        closest3_babai = babai_rounding_method(B3, v3)
        closest3_kannan = decode_kannan_embedding(*kannan_embedding(B3, v3))
        self.assertTrue(np.allclose(closest3_babai, v3))
        self.assertTrue(np.allclose(closest3_kannan, v3))

        # Test case 4: Close to a known combination
        v4 = B3 @ (coeffs3 + np.array([0.1, -0.2, 0.3, 0.1, -0.1]))
        closest4_babai = babai_rounding_method(B3, v4)
        closest4_kannan = decode_kannan_embedding(*kannan_embedding(B3, v4))
        self.assertTrue(np.allclose(closest4_babai, v3))
        self.assertTrue(np.allclose(closest4_kannan, v3))


    def test_babai_vs_kannan(self):
        """Babai and Kannan should give (nearly) identical results."""
        for _ in range(10):  # Run several random tests
            dim = np.random.randint(2, 10)
            B = generate_better_lattice(dim)
            coeffs = np.random.uniform(-5, 5, size=dim)
            v = B @ coeffs

            reduced_B = lll_reduction(B)  # Reduce the basis

            closest_babai = babai_rounding_method(reduced_B, v)
            closest_kannan = decode_kannan_embedding(*kannan_embedding(reduced_B, v))

            # Check that the *distances* are very close (not necessarily the vectors)
            self.assertAlmostEqual(np.linalg.norm(closest_babai - v),
                                   np.linalg.norm(closest_kannan - v),
                                   places=5)  # Allow for small numerical differences

    def test_lll_reduction_properties(self):
        """Test properties of the LLL-reduced basis."""
        for _ in range(10):
            dim = np.random.randint(2, 10)
            B = generate_better_lattice(dim)
            reduced_B = lll_reduction(B)

            # 1. Check that the reduced basis spans the same lattice.
            #    The determinant of the transformation matrix should be +/- 1.
            transformation_matrix = np.linalg.solve(B, reduced_B)
            self.assertAlmostEqual(abs(np.linalg.det(transformation_matrix)), 1, places=5)

            # 2. Check that the reduced basis vectors are shorter (on average).
            original_norms = np.linalg.norm(B, axis=0)
            reduced_norms = np.linalg.norm(reduced_B, axis=0)
            self.assertTrue(np.mean(reduced_norms) <= np.mean(original_norms) * (2**(dim/4) )) # theoretical bound


            # 3. (Optional, more advanced) Check for reduced basis properties
            #    (e.g., Lovasz condition, size reduction condition).  This
            #    requires reimplementing parts of the LLL algorithm within
            #    the test, which is generally not recommended unless you're
            #    debugging the LLL implementation itself.

    def test_shortest_vector_nonzero(self):
        """Ensure the shortest vector found is not the zero vector."""
        for _ in range(10):
            dim = np.random.randint(2, 10)
            B = generate_better_lattice(dim)
            reduced_B = lll_reduction(B)
            shortest_vector = shortest_vector_search(reduced_B)  # Use a reasonable search_radius
            self.assertFalse(np.allclose(shortest_vector, np.zeros(dim)))

    def test_exhaustive_search_small_dimensions(self):
        """Compare Babai/Kannan with exhaustive search (for small dimensions)."""
        for dim in range(2, 5):  # Only for very small dimensions
            for _ in range(5): # multiple tests for each dimension
                B = generate_better_lattice(dim)
                coeffs = np.random.uniform(-2, 2, size=dim) # smaller range
                v = B @ coeffs
                reduced_B = lll_reduction(B)

                closest_babai = babai_rounding_method(reduced_B, v)
                closest_kannan = decode_kannan_embedding(*kannan_embedding(reduced_B, v))
                closest_exhaustive = exhaustive_closest_vector_search(reduced_B, v, search_radius=5) # smaller search_radius

                # Check that Babai/Kannan are at least as good as exhaustive
                self.assertLessEqual(np.linalg.norm(closest_babai - v),
                                    np.linalg.norm(closest_exhaustive - v) + 1e-8)  # Allow for tiny numerical errors
                self.assertLessEqual(np.linalg.norm(closest_kannan - v),
                                    np.linalg.norm(closest_exhaustive - v) + 1e-8)

if __name__ == '__main__':
    unittest.main()