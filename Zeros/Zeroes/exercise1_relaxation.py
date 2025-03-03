"""Relaxation method: we have to rearrange eq. as x = f(x)"""
"""Exercise: solve for 2 - e^-x = x: make a plot to see the number of roots and then use relaxation method to find them"""
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    f = 2 - np.exp(-x)
    return f

def f2(x):
    f = - np.log(2-x)
    return f


def relaxation(func, tol):
    x = 1
    xold = 10.0
    while(abs(x - xold) > tol):
        xold = x
        x = func(x)
    return x

tol = 1e-6

#plotting
x = np.arange(-2, 3, 0.1)
plt.plot(x, f(x) - x, label="y = f(x)")
plt.axhline(y=0, color="green", linestyle="--", label="y = 0")  #plot di linea verticale su y = 0 per vedere facilmente gli zeri
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.show()

#finding the roots
roots = []
roots.append(relaxation(f, tol))
print("First root: converged to x =", roots[0])

"""Since |df/dx| > 1 in the negative solution, the relaxation algorithm does 
# not converge there. Need to find a suitable transformation and write the
# equation in the form g(x) = x, where |dg/dx| < 1.
# This can be done by rearranging the equation as  x = - log(2-x)"""

roots.append(relaxation(f2, tol))
print("First root: converged to x =", roots[1])

#plotting
plt.plot(x, f(x) - x, label="y = f(x)")
plt.axhline(y=0, color="green", linestyle="--", label="y = 0")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.axvline(x=roots[0], color="red", linestyle="--", label="root 1")
plt.axvline(x=roots[1], color="purple", linestyle="--", label="root 2")
plt.legend()
plt.show()