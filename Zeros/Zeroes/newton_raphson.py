"""" Finds a solution of func(x)=0 using Newton's method
    deriv is an input function which computes df/dx
    x0 is the starting point of the search
    #tol is the tolerance, set by default at 1e-4"""

import numpy as np

def newton(func, deriv, x0, tol=1e-4):
    x = x0
    acc = tol + 1.

    while (acc > tol):
        delta = func(x) / deriv(x)
        x1 = x - delta
        acc = abs(x1 - x)
        x = x1

    return x

def f(x):
    func = 2 - x - np.exp(-x)
    return func

def deriv(x):
    d = -1 + np.exp(-x)
    return d

x0 = 1
print(newton(f, deriv, x0, tol=1e-4))
x0=-1
print(newton(f, deriv, x0, tol=1e-4))
