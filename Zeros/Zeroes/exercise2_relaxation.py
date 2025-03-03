"""Solve x = 1 - e^(-cx) for c from 0 to 3 in steps of 0.01 and plot x(c) (roots as function of c)"""
import numpy as np
import matplotlib.pyplot as plt

def relaxation(func, tol):
    x = 1
    xold = 10.0
    while(abs(x - xold) > tol):
        xold = x
        x = func(x)
    return x

# Input function
def func(x,c):
    f = 1 - np.exp(-c*x)
    return f

# Problem parameters
start = 10.0
tol = 1e-6
cmin = 0.
cmax = 3.
step = 0.01

roots = [ ]

for c in np.arange(cmin,cmax,step):

    # Calling function func for fixed parameter c
    # To be passed to function relax
    f = lambda x: func(x,c)
    # Finding root
    root = relaxation(f,tol)
    # Updating list of roots for varying c
    roots.append(root)

print(roots)

plt.plot(np.arange(cmin,cmax,step),roots)
plt.xlabel('c', size = 16)
plt.ylabel('x(c)', size = 16)
plt.show()