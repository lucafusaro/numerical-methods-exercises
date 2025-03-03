import numpy as np

"""For linear system ( jacobian matrix coincide con A):"""
def newton_raphson(A, b, x0, tolerance=1e-6, max_iterations=100):
    """
    Solve a linear system Ax = b using the Newton-Raphson method.

    Parameters:
    - A: Coefficient matrix
    - b: Right-hand side vector
    - x0: Initial guess for the solution
    - tolerance: Convergence criteria (default is 1e-6)
    - max_iterations: Maximum number of iterations (default is 100)

    Returns:
    - x: Solution vector
    - iterations: Number of iterations performed
    """

    n = len(x0)
    x = x0.copy()

    for iteration in range(max_iterations):
        # Calculate the residual and Jacobian matrix
        residual = A.dot(x) - b
        J = A

        # Update the solution using the Newton-Raphson formula
        delta_x = np.linalg.solve(J, -residual)
        x += delta_x

        # Check for convergence
        if np.linalg.norm(delta_x) < tolerance:
            return x, iteration + 1

    # If the method does not converge, raise an exception
    raise RuntimeError("Newton-Raphson method did not converge within the specified number of iterations")

# Example usage:
A = np.array([[2, -1], [1, 1]])
b = np.array([3, 4])
x0 = np.array([0, 0], dtype=float)

solution, iterations = newton_raphson(A, b, x0)
print("Solution:", solution)
print("Iterations:", iterations)

"""Non linear system:"""

import numpy as np

def nonlinear_system_newton_raphson(f, J, x0, tolerance=1e-6, max_iterations=100):
    """
    Solve a nonlinear system of equations using the Newton-Raphson method.

    Parameters:
    - f: Function representing the system of equations (returns a vector)
    - J: Jacobian matrix of the system of equations
    - x0: Initial guess for the solution
    - tolerance: Convergence criteria (default is 1e-6)
    - max_iterations: Maximum number of iterations (default is 100)

    Returns:
    - x: Solution vector
    - iterations: Number of iterations performed
    """

    x = x0.copy()

    for iteration in range(max_iterations):
        # Calculate the residual and Jacobian matrix
        residual = f(x)
        J_matrix = J(x)

        # Update the solution using the Newton-Raphson formula
        delta_x = np.linalg.solve(J_matrix, -residual)
        x += delta_x

        # Check for convergence
        if np.linalg.norm(delta_x) < tolerance:
            return x, iteration + 1

    # If the method does not converge, raise an exception
    raise RuntimeError("Newton-Raphson method did not converge within the specified number of iterations")

# Define the system of equations
def f(x):
    return np.array([x[0]**2 + x[1]**2 - 4, x[0] - x[1] - 1])

# Define the Jacobian matrix
def J(x):
    return np.array([[2*x[0], 2*x[1]], [1, -1]])      # [[df1/dx, df1/dy], [df2/dx, df2/dy]]

# Example usage:
x0 = np.array([1.0, 1.0])

solution, iterations = nonlinear_system_newton_raphson(f, J, x0)
print("Solution:", solution)
print("Number of Iterations:", iterations)

