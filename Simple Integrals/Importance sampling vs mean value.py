import numpy as np
import matplotlib.pyplot as plt
"""Integrand:"""
def f(x):
    funz = x**(-1/2) / (np.exp(x) + 1)   #da integrare tra 0 e 1 come richiesto nella traccia
    return funz
"""Choosing a weight function:"""
def weight_function(x):
    func = x**(-1/2)
    return func
"Normalize to find pdf p(x)"
#integral of w(x) between 0 and 1 is equal to 2, so define p(x)
def p(x):
    return weight_function(x) / 2
"""Inverse random sampling:"""
def y(x):
    return x**2

"""For comparison with mean value method:"""
def mean_value(N, a_x, b_x):    #N number of points generated, a and b: extremes of interval
    sum_values = 0
    for i in range(int(N)):
        x = np.random.uniform(a_x, b_x)
        sum_values += f(x)
    I = ((b_x - a_x) / N) * sum_values
    return I


N_values = [1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6]
I_values_impsam = []
I_values_meanv = []
for N in N_values:
    y_values = []
    for i in range(int(N)):
        x = np.random.random()
        y_values.append(y(x))

    sum_values = 0
    for y_i in y_values:
        sum_values += f(y_i) / weight_function(y_i)

    I = (2 / N) * sum_values
    I_values_impsam.append(I)
    print("Integrale calcolato con N = " + str(int(N)) + " punti: I =", I)
    "For comparison with mean value method:"
    I_values_meanv.append(mean_value(N, 0, 1))

"Plot I vs N and comparison between methods:"
plt.plot(N_values, I_values_impsam, marker='o', color="red", label="Importance sampling")
plt.plot(N_values, I_values_meanv, marker='o', color="green", label="Mean value")
plt.xlabel("N")
plt.xscale("log")
plt.ylabel("I")
plt.legend()
plt.show()



