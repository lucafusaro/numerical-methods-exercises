"""Exercise: assuming circular orbits and that the Earth is much more massive than the moon,
 find the L1 point of a satellite orbiting between the two
Show that, in L1:  𝐺𝑀 / r^2 − 𝐺 𝑚 / (𝑅−𝑟)^2 = 𝜔^2 * 𝑟 ,
 where  𝑅 is the moon-earth distance, M and m are the respective masses,  𝐺 is Newton's constant
 and 𝜔 is the angular velocity of both the moon and the satellite
 Write a program that solves the equation above with at least four significant figures,
  using Newton's or secant's method. Parameters (SI units):  𝐺 = 6.674×10^-11
 𝑀 = 5.974×10^24 , 𝑚 = 7.348×10^22, 𝑅 = 3.844×10^8, 𝜔 = 2.662×10^-6
Make a plot to verify that your code computed the correct solution"""

import numpy as np
import sys
import matplotlib.pyplot as plt

#parameters
G = 6.674e-11
M = 5.974e24
m = 7.348e22
R = 3.844e8
omega = 2.662e-6

def f(r):
    func = (G * M / r**2) - (G * m / (R - r)**2) - omega**2 * r
    return func

def deriv(r):
    der = (-2 * G * M / r**3) - (2 * G * m / (R - r)**3) - omega**2
    return der

def newton(func, deriv, x0, tol=1e-4):
    x = x0
    acc = tol + 1.

    while (acc > tol):
        delta = func(x) / deriv(x)
        x1 = x - delta
        acc = abs(x1 - x)
        x = x1

    return x

x0 = (2 / 3) * R
L1 = newton(f, deriv, x0)
print("Position of L1 in unit of R:", L1 / R )  #L1 si trova tra la terra e la luna --> L1 < R

# Plotting
pos = np.arange(0.5 * R, 0.95 * R, 0.01 * R)
x = np.arange(0.5, 0.95, 0.01)
fx = f(pos)
plt.axhline(linestyle='dashed', color='red', linewidth=1)
plt.axvline(x=L1 / R, linestyle='dashed', color='red', linewidth=1)
plt.xlabel('x/R', size=16)
plt.ylabel('f(x)', size=16)
plt.annotate('$L_1$', xy=(2, 0.), xycoords='axes points', xytext=(270, -15), color='red', size=14)
plt.plot(x, fx)
plt.show()

