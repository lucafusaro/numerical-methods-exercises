"""Write a script to implement relaxation method, bisection method and Newton-Raphson.
Use them to find the zeros of the following function of the eccentric anomaly E:
F = E - ecc * np.sin(E) where ecc is the eccentricity, solve for ecc = 0.1, 0.7, 0.9.
F is the mean anomaly, use F = pi and pi/3. E is the eccentric anomaly,
and it's the unknown of the exercise. Use tolerance 10^-6."""

import numpy as np
import sys

"Function for bisection method and Newton-Raphson( f(x) = 0):"
def func2(E, ecc, F):
    return E - ecc * np.sin(E) - F

def f(E):
    return func2(E, ecc, F)

"Derivative of f(x) for Newton-Raphson:"
def deriv(E):
    return 1 - ecc * np.cos(E)

"Function for relaxation method x = f(x):"
def f2(E, ecc, F):
    return F + ecc * np.sin(E)

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

def relaxation(tol):
    x = 1.0
    xold = 10.0
    while(abs(x - xold) > tol):
        xold = x
        #x = 2.0 - np.exp(-x)
        x = f2(x, ecc, F)
        #print(x)
    return x

def newton(func, deriv, x0, tol=1e-4):
    x = x0
    acc = tol + 1.

    while (acc > tol):
        delta = func(x) / deriv(x)
        x1 = x - delta
        acc = abs(x1 - x)
        x = x1
    return x


#choose interval
E1 = 1
E2 = 5
#start for Newton-Raphson
start = np.pi /2
#accuracy tolerance
epsilon = 1e-6
#parameters of function
ecc_values = [0.1, 0.7, 0.9]
F_values = [np.pi / 3, np.pi]
for F in F_values:
    for ecc in ecc_values:
        print("\nFor ecc =", str(ecc), "and F =", str(F), "Converged to:")
        print("Bisection method: E = ", bisection(f, E1, E2, epsilon))
        print("Relaxation method: E = ", relaxation(epsilon))
        print("Newton-Raphson method: E = ", newton(f, deriv, start, tol=epsilon))



