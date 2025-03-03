""""Integrate sin(1 / (x * (2 - x))) ** 2 between 0 and 2, calculate the integral with
N = 10^3, 5 * 10^3, 10^4, 5 * 10^4, 10^5, 5 * 10^5, 10^6, 5 * 10^6.
Estimate and plot the difference you obtain by repeating this exercise
with different values of N ( plot I vs N)"""

import numpy as np
import matplotlib.pyplot as plt
def integrand(x):
    func = (np.sin(1 / (x * (2 - x)))) ** 2
    return func

def mean_value(integrand, a, b, N):    #N number of points generated, a and b: extremes of interval
    sum_values = 0
    for i in range(int(N)):
        x = np.random.uniform(a, b)
        sum_values += integrand(x)
    I = ((b - a) / N) * sum_values
    return I

"""Plot function"""
x_values = np.arange(0.001, 2, 0.001)
vectorized_integrand = np.vectorize(integrand)
plt.plot(x_values, vectorized_integrand(x_values))
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Integrand function")
plt.show()

"""Computation for different N:"""
N_values = [10**3, 5 * 10**3, 10**4, 5 * 10**4, 10**5, 5 * 10**5, 10**6, 5 * 10**6]
I_values = []
j = 0
for N in N_values:
    I_values.append(mean_value(integrand, 0, 2, N))
    print("Il risultato ottenuto generando", N, "punti è: I =", I_values[j])
    j += 1

#Remember: the error scales like N^(-1/2), the mean value method has a lower error respect monte carlo
#Nota: questo metodo è molto utile anche per integrali in più dimensioni, in questo caso I = (V/ N) * sum(f(r_i))
# dove r_i sono punti generati casualmente nel volume V. Questo metodo ha però problemi per integrali patologici

"""Plot I vs N:"""
plt.plot(N_values, I_values, marker='o')
plt.ylabel('I')
plt.xlabel('N')
plt.xscale('log', base=10)
plt.show()
