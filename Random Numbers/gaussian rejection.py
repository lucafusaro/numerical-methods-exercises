"""Generate N = 10^4 points distributed according to a Gaussian (with sigma = 2
and centered around zero) with the rejection method, by using the
distribution function f(x) = 1, uniform between min = -50 and max = 50."""

import numpy as np
import matplotlib.pyplot as plt
def gaussian(x, sigma=2):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(1/2)*(x/sigma)**2)

#Define the range for x values
x_min, x_max = -50, 50

# Calculate y_max, the integral of f(x)=1 (uniform distribution) from x_min to x_max
y_max = (x_max - x_min)

#number of points
N = 1e5     #nota: usando N = 1e4 points come da traccia non si ottiene un buon risultato

y_values = []
x_values = []
for i in range(int(N)):
    y = np.random.uniform(0, y_max)
    x = y - 50    #y = g(x) = x + 50 --> x = y - 50
    m = np.random.uniform(0, 1)
    if m <= gaussian(x):
        x_values.append(x)

"Plotting"
plt.hist(x_values, bins=40, density=True, label='Generated points')
x_point = np.linspace(x_min, x_max, 1000)
plt.ylim(top=0.3)
plt.plot(x_point, gaussian(x_point))
plt.show()
