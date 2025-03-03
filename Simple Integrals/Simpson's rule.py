"""Write your own function for Simpson's integration.
 The function should double the number of points in the
  sampled interval until a specifified level of accuracy is reached"""

def simp(func, a, b, tol):
    demomode = False

    # Initial number of sampled points
    N = 10

    # 1/3*f(a) + 1/3*f(b)
    s = (1. / 3.) * func(a) + (1. / 3.) * func(b)

    # Starting with N points
    h = (b - a) / N
    s1odd = 0.
    for k in range(1, N, 2):
        s1odd += func(a + k * h)
    s1even = 0.
    for k in range(2, N, 2):
        s1even += func(a + k * h)

    # Simpson's rule with N points in [a,b]
    simp1N = h * (s + (4. / 3.) * s1odd + (2. / 3.) * s1even)

    if demomode:
        print(N, simp1N)

    acc = tol + 1.

    while (acc > tol):
        # Now doubling the number of points and updating h
        N = 2 * N
        h = (b - a) / N
        # All points in previous interval (odd+even) are
        # now the even points in the new interval.
        s2even = s1odd + s1even
        # Now getting s2odd
        s2odd = 0.
        for k in range(1, N, 2):
            s2odd += func(a + k * h)
        # Simpson's rule with 2N points in the sample
        simp2N = h * (s + (4. / 3.) * s2odd + (2. / 3.) * s2even)

        acc = ((1. / 15.) * abs(simp2N - simp1N)) / abs(simp2N)

        # Update and loop
        s1even, s1odd = s2even, s2odd
        simp1N = simp2N

        if demomode:
            print(N, simp2N)

        # Exit loop if N > 1e6
        if (N > 1e6):
            print(' ')
            print('The required accuracy could not be reached with N=1e6 points.')
            print('Stopping here, N =', N, '; acc =', acc)
            break

    return simp2N


"""Example using own Simpson function and scipy simpson"""

import numpy as np
import scipy.integrate as integrate
from matplotlib import pyplot as plt


# Define integrand
def integrand(theta, x, m):
    integrand = (1.0 / np.pi) * np.cos(m * theta - x * np.sin(theta))
    return integrand


#using own function
def bessel(x, m):
    integ = lambda theta: integrand(theta, x, m)
    tol = 1e-4
    bes = simp(integ, 0., np.pi, tol)
    return bes


# Calculate Bessel using Simpson rule from scipy
N = 100

#using scipy
def bessel_simp(x, m, N):
    dtheta = np.pi / float(N)
    grid = np.arange(0.0, np.pi, dtheta)
    y = integrand(grid, x, m)
    bes = integrate.simpson(y, grid)

    return bes


x = 3
m = 2
bes = bessel(x, m)
bes2 = bessel_simp(x, m, N)
print(bes)

xlist = np.arange(0, 20, 0.1)
for m in range(3):

    bes = []
    bes2 = []

    for x in xlist:
        b1 = bessel(x, m)
        b2 = bessel_simp(x, m, N)
        bes.append(b1)
        bes2.append(b2)

    plt.plot(xlist, bes, label='Simpson, my function')
    plt.plot(xlist, bes2, label='Simpson, scipy')
    plt.xlabel('x', fontsize=16)
    plt.ylabel('$J_m(x)$', fontsize=16)
    plt.legend()
    plt.title('Bessel function, m = ' + str(m), fontsize=16)
    plt.show()