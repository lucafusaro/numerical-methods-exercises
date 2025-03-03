import numpy as np

def mean_value_method_3d(f, x_range, y_range, z_range):
    """
    Implements the mean value method for 3D integration.

    Parameters:
    - f: The function to be integrated.
    - x_range, y_range, z_range: Tuples representing the ranges of integration for each dimension.

    Returns:
    - Integral approximation.
    """
    # Generate random sample points within the specified ranges
    num_samples = 100000
    x_samples = np.random.uniform(x_range[0], x_range[1], num_samples)
    y_samples = np.random.uniform(y_range[0], y_range[1], num_samples)
    z_samples = np.random.uniform(z_range[0], z_range[1], num_samples)

    # Evaluate the function at the sample points
    f_values = f(x_samples, y_samples, z_samples)

    # Compute the volume of the integration region
    volume = (x_range[1] - x_range[0]) * (y_range[1] - y_range[0]) * (z_range[1] - z_range[0])

    # Compute the mean value of the function
    mean_value = np.mean(f_values)

    # Approximate the integral using the mean value
    integral_approximation = mean_value * volume

    return integral_approximation

# Example: Compute the integral of the function f(x, y, z) = x^2 + y^2 + z^2 over the unit cube [0, 1] x [0, 1] x [0, 1]
def example_function(x, y, z):
    return x**2 + y**2 + z**2

x_range = (0, 1)
y_range = (0, 1)
z_range = (0, 1)

result = mean_value_method_3d(example_function, x_range, y_range, z_range)

print("Approximated Integral:", result)

"With accuracy"
import numpy as np

def mean_value_method_3d(f, x_range, y_range, z_range, accuracy=1e-4):
    """
    Implements the mean value method for 3D integration.

    Parameters:
    - f: The function to be integrated.
    - x_range, y_range, z_range: Tuples representing the ranges of integration for each dimension.
    - accuracy: Desired accuracy of the integral approximation (default: 1e-6).

    Returns:
    - Integral approximation.
    """
    num_samples = 100
    integral_approximation_prev = float('inf')

    while True:
        x_samples = np.random.uniform(x_range[0], x_range[1], num_samples)
        y_samples = np.random.uniform(y_range[0], y_range[1], num_samples)
        z_samples = np.random.uniform(z_range[0], z_range[1], num_samples)

        f_values = f(x_samples, y_samples, z_samples)

        volume = (x_range[1] - x_range[0]) * (y_range[1] - y_range[0]) * (z_range[1] - z_range[0])

        mean_value = np.mean(f_values)

        integral_approximation = mean_value * volume

        if np.abs(integral_approximation - integral_approximation_prev) < accuracy:
            break

        integral_approximation_prev = integral_approximation
        num_samples *= 2

    return integral_approximation

# Example: Compute the integral of the function f(x, y, z) = x^2 + y^2 + z^2 over the unit cube [0, 1] x [0, 1] x [0, 1]
def example_function(x, y, z):
    return x**2 + y**2 + z**2

x_range = (0, 1)
y_range = (0, 1)
z_range = (0, 1)

result = mean_value_method_3d(example_function, x_range, y_range, z_range, accuracy=1e-6)

print("Approximated Integral:", result)

