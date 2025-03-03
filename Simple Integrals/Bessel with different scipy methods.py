import  numpy as np
import scipy.integrate as integrate
from matplotlib import pyplot as plt

"""Diffraction pattern  𝐼(𝑟)=(J1(𝑘𝑟)/𝑘𝑟)^2
   By definition  J𝑚(𝑥) = (1/ pi ) ∫𝑐𝑜𝑠(𝑚Θ−𝑥𝑠𝑖𝑛Θ)𝑑Θ between 0 and pi
a) Write a python function that calculates  J𝑚(𝑥), use it to make a plot of J0, J1, J2
  in the x-interval [0,20]

b) Make a density plot of the intensity of the diffraction pattern, for a point light source with  𝜆=500
  nm in a square region of the focal plane with r in the range [0,1] 𝜇m"""

#define integrand
def integrand(theta, x, m):
    f = (1 / np.pi) * np.cos(m * theta - x * np.sin(theta))
    return f

#Calculate Bessel using general purpose integration routine from python
def bessel(func,x,m):
    bes = integrate.quad(func,0.0,np.pi,args=(x,m))
    return bes
# Calculate Bessel using Romberg
def bessel_rom(func,x,m):
    bes = integrate.romberg(func,0.0,np.pi,args=(x,m))
    return bes
# Calculate Bessel using trapezoidal rule
def bessel_trap(x,m,N):
    dtheta = np.pi/float(N)
    grid = np.arange(0.0,np.pi,dtheta)
    y = integrand(grid,x,m)
    bes = integrate.trapezoid(y,grid)

    return bes
# Calculate Bessel using Simpson rule
def bessel_simp(x,m,N):
    dtheta = np.pi/float(N)
    grid = np.arange(0.0,np.pi,dtheta)
    y = integrand(grid,x,m)
    bes = integrate.simpson(y,grid)

    return bes
# Just a quick check that different methods
# give consistent answers in a given x, for given m
bes1 = bessel_simp(3,2,10000)
bes2 = bessel_rom(integrand,3,2)
print(bes1,bes2)

xlist = np.arange(0, 20, 0.1)
N = 1000

for m in range(3):

    bes = []
    bes_rom = []
    bes_trap = []
    bes_simp = []

    for x in xlist:
        b1 = (bessel(integrand, x, m))
        bes.append(b1[0])
        bes_rom.append(bessel_rom(integrand, x, m))
        bes_trap.append(bessel_trap(x, m, N))
        bes_simp.append(bessel_simp(x, m, N))

    plt.plot(xlist, bes_trap, label='trapezoid, N =' + str(N))
    plt.plot(xlist, bes_simp, label='Simpson, N=' + str(N))
    plt.plot(xlist, bes_rom, label='Romberg')
    plt.plot(xlist, bes, label='Python quad')
    plt.xlabel('x', fontsize=16)
    plt.ylabel('$J_m(x)$', fontsize=16)
    plt.legend()
    plt.title('Comparison between methods, m = ' + str(m), fontsize=16)
    plt.show()

