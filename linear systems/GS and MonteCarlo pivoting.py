import numpy as np


def gauss_seidel_with_pivoting(A, b, x0=None, tolerance=1e-3, max_iterations=1000):
    n = len(b)
    x = np.zeros_like(b) if x0 is None else x0.copy()

    for iteration in range(max_iterations):
        x_old = x.copy()

        for i in range(n):
            # Partial pivoting: find the row with the maximum absolute value in the current column
            pivot_row = np.argmax(np.abs(A[i:, i])) + i
            # Swap the current row with the row containing the maximum element
            A[[i, pivot_row]] = A[[pivot_row, i]]
            b[i], b[pivot_row] = b[pivot_row], b[i]

            sigma = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i + 1:], x_old[i + 1:])
            x[i] = (b[i] - sigma) / A[i, i]

        # Check for convergence
        if np.linalg.norm(x - x_old) < tolerance:
            return x, iteration + 1

    raise RuntimeError("Gauss-Seidel method with pivoting did not converge within the specified number of iterations.")


def monte_carlo_pivoting(A, b, max_iterations=100):
    n = len(b)

    for _ in range(max_iterations):
        # Randomly permute the equations
        permuted_indices = np.random.permutation(n)
        A = A[permuted_indices]
        b = b[permuted_indices]

        # Attempt to solve the system with the new permutation
        try:
            solution, _ = gauss_seidel_with_pivoting(A, b)
            return solution
        except RuntimeError:
            pass  # Continue with the next iteration if the Gauss-Seidel method does not converge

    raise RuntimeError("Monte Carlo pivoting did not converge within the specified number of iterations.")


# Solve the system with Monte Carlo pivoting
solution_monte_carlo = monte_carlo_pivoting(A, b)

print("Solution using Gauss-Seidel with Monte Carlo pivoting:")
print("Solution:", solution_monte_carlo)