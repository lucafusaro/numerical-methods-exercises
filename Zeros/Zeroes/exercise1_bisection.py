"""Solve 2 - e^(-x) = x with bisection method"""

import numpy as np
import sys
import matplotlib.pyplot as plt

def bisection(f, x1, x2, accuracy):
    while abs(x1 - x2) > accuracy:
        if f(x1) > 0 > f(x2) or f(x1) < 0 < f(x2):
            #mid point
            x = 0.5 * (x1 + x2)
            if f(x) * f(x1) > 0: #same sign
                x1 = x
            else:
                x2 = x
        else:
            sys.exit('f(x) does not have opposite signs at the boundaries')
        x = 0.5 * (x1 + x2)
    return x

def f(x):
    f = 2.0 - np.exp(-x) -x
    return f

#plotting
x = np.arange(-3, 3, 0.1)
plt.plot(x, f(x), label="y = f(x)")
plt.axhline(y=0, color="green", linestyle="--", label="y = 0")  #plot di linea verticale su y = 0 per vedere facilmente gli zeri
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.show()

#accuracy tolerance
epsilon = 1e-6
#choose interval
#first root
x1 = -2
x2 = 0
root1 = bisection(f, x1, x2, epsilon)
print("La prima radice è x =",root1)
#second root
x1 = 0
x2 = 2
root2 = bisection(f, x1, x2, epsilon)
print("La seconda radice è x =",root2)

#plot solution
roots = [root1, root2]
plt.plot(x, f(x), label="y = f(x)")
plt.axhline(y=0, color="green", linestyle="--", label="y = 0")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.axvline(x=roots[0], color="red", linestyle="--", label="root 1")
plt.axvline(x=roots[1], color="purple", linestyle="--", label="root 2")
plt.legend()
plt.show()
