""". Write a script that uses the Newton-Raphson method to solve the following
system of two non-linear equations
sin (x y) + x = 2
y^2 − x y = 3
Start by assuming x = 1 and y = 1, to find the closest solution to such values.
Suggestions:
1. Remember that the Newton-Raphson method is a root finding method.
2. Use pen and paper to write down the partial derivatives and the Jacobian matrix before you
start coding.
3. As demonstrated during the lectures, the Newton-Raphson method applied to a system of non-linear equations
 requires to iteratively solve a system of linear equations. Use
numpy.linalg.solve to solve this system of linear equations."""

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
    return np.array([np.sin(x[0]*x[1]) + x[0] - 2, x[1]**2 - x[0]*x[1] - 3])

# Define the Jacobian matrix
def J(x):
    return np.array([[np.cos(x[0]*x[1]) * x[1] + 1, np.cos(x[0]*x[1]) * x[0]], [- x[1], 2*x[1] - x[0]]])

# Example usage:
x0 = np.array([1.0, 1.0])

solution, iterations = nonlinear_system_newton_raphson(f, J, x0)
print("Solution:", solution)
print("Number of Iterations:", iterations)