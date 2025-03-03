""""Integrate sin(1 / (x * (2 - x))) ** 2 between 0 and 2, calculate the integral with
N = 10^3, 5 * 10^3, 10^4, 5 * 10^4, 10^5, 5 * 10^5, 10^6, 5 * 10^6.
Estimate and plot the difference you obtain by repeating this exercise
with different values of N ( plot I vs N)"""

import numpy as np
import matplotlib.pyplot as plt


def integrand(x):
    func = (np.sin(1 / (x * (2 - x)))) ** 2
    return func
def monte_carlo(N, a_x, b_x, a_y, b_y):    #N number of points generated, a and b: extremes of interval
    k = 0  # k = number of points below the curve
    A = (b_x - a_x) * (b_y - a_y)     #area of the interval
    for i in range(int(N)):
        x = np.random.uniform(a_x, b_x)
        y = np.random.uniform(a_y, b_y)
        if y <= integrand(x):
            k += 1
    I = (k/N) * A   #l'integrale si può considerare come il rapporto tra k ed N per l'area della regione
    return I

"""Plot function"""
x_values = np.arange(0.001, 2, 0.001)
vectorized_integrand = np.vectorize(integrand)
plt.plot(x_values, vectorized_integrand(x_values))
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Integrand function")
plt.show()

"""Computation of integral for different N:"""
N_values = [10**3, 5 * 10**3, 10**4, 5 * 10**4, 10**5, 5 * 10**5, 10**6, 5 * 10**6]
I_values = []
j = 0
for i in N_values:
    I_values.append(monte_carlo(i, 0, 2, 0, 1))
    print("Il risultato ottenuto generando", i, "punti è: I =", I_values[j])
    j += 1
#Remember: the error scales like N^(-1/2)

"""Plot I vs N:"""
plt.plot(N_values, I_values, marker='o')
plt.ylabel('I')
plt.xlabel('N')
plt.xscale('log', base=10)
plt.show()


"Monteccarlo 3D with accuracy"
import numpy as np


def monte_carlo_integration_3d(f, x_range, y_range, z_range, num_samples=1000, accuracy=1e-6):
    """
    Implements the Monte Carlo method for 3D integration.

    Parameters:
    - f: The function to be integrated.
    - x_range, y_range, z_range: Tuples representing the ranges of integration for each dimension.
    - num_samples: Number of random samples (default: 1000).
    - accuracy: Desired accuracy of the integral approximation (default: 1e-6).

    Returns:
    - Integral approximation.
    """
    volume = (x_range[1] - x_range[0]) * (y_range[1] - y_range[0]) * (z_range[1] - z_range[0])

    integral_approximation_prev = float('inf')

    while True:
        x_samples = np.random.uniform(x_range[0], x_range[1], num_samples)
        y_samples = np.random.uniform(y_range[0], y_range[1], num_samples)
        z_samples = np.random.uniform(z_range[0], z_range[1], num_samples)

        f_values = f(x_samples, y_samples, z_samples)

        points_below_surface = np.count_nonzero(z_samples < f_values)

        integral_approximation = points_below_surface / num_samples * volume

        if np.abs(integral_approximation - integral_approximation_prev) < accuracy:
            break

        integral_approximation_prev = integral_approximation
        num_samples *= 2

    return integral_approximation


# Example: Compute the integral of the function f(x, y, z) = x^2 + y^2 + z^2 over the unit cube [0, 1] x [0, 1] x [0, 1]
def example_function(x, y, z):
    return x ** 2 + y ** 2 + z ** 2


x_range = (0, 1)
y_range = (0, 1)
z_range = (0, 1)

result = monte_carlo_integration_3d(example_function, x_range, y_range, z_range, accuracy=1e-6)

print("Approximated Integral:", result)
