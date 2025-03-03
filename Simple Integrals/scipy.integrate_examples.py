"""How to use trapz function of scipy.integrate; 
(1) Methods for integrating functions given fixed samples:
The python function takes as argument an array of points f(xi)
(i.e. a fixed sample) and the corresponding array xi over which
it performs the integration. Same as this lecture.
Functions
trapz: trapezoid rule
cumtrapz: cumulative integral with trapezoidal rule
simps: Simpson’s rule
romb: Romberg integration"""

from scipy.integrate import trapz, quad, cumtrapz, cumulative_trapezoid
import numpy as np
"""Exercise of NFW profile"""
G = 6.667e-8 #gravity const in cgs
pc = 3.086e18 #1 pc in cgs
msun = 1.989e33 #solar mass in cgs

rs = 10.0 * 1e3 * pc #10 kpc in cgs
rmax = 100.0 * rs
rho0 = 1e8 * msun / (1e3 * pc) ** 3 #1e8 Msun/kpc^3 in cgs

def NFW(r):
    x = r/rs
    rho = rho0 / ((1. + x) ** 2)
    mass = 4. * np.pi * rs * r * rho
    return mass

b = rmax
a = 0.0
intervallo = (b - a)
N = int(1e6)
h = intervallo / N

trapzx = [a]
trapzy = [NFW(a)]
for i in range(1, N-1, 1):
    trapzx.append(a + i*h)
    trapzy.append(NFW(a + i*h))
trapzx.append(b)
trapzy.append(NFW(b))

I = trapz(trapzy, trapzx)
print("With scipy.integrate.trapz I =", I/msun, "Msun")

"""(2) Methods for integrating functions given function object:
The python function takes the function you want to integrate as an argument, 
plus the integration range, and decides in which points xi to evaluate the function. 
This is exactly what we have done for the look-backtime in the example 
examples/python/lookback.py
Functions
quad: General purpose integration 
(uses a technique from the Fortran library QUADPACK: 
contains different algorithms to solve integrals)
dblquad: General purpose integration in two dimensions (two variables)
tplquad: General purpose integration in three dimensions (three variables)
fixed_quad: Integrate func(x) using Gaussian quadrature of order N
quadrature: Integrate with given tolerance using Gaussian quadrature 
romberg: Integrate func(x) using Romberg integration """

#Example
OmegaM = 0.2726  #omega matter, parameter from cosmology
OmegaL = 0.7274  #omega lambda, parameter from cosmology

def integrand(x):
    r = 1. / ((1 + x) * (OmegaM * (1. + x)**3. + OmegaL)**0.5)
    return r
#we can also use compact form with lambda function
z = 200
I = quad(integrand, 0.0, z, epsrel= 1e-13) #epsrel: max relative integration error tolerated by quad
"Nota quad ritorna due valori, integrale ed errore, usa [0] o [1] per ottenere solo uno dei due"
print(I)
