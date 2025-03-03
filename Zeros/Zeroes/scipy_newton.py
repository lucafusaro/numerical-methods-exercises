"""Exercise on lagrangian point with scipy newton method"""

from scipy import optimize as opt
import matplotlib.pyplot as plt
import numpy as np

# Parameters
G = 6.674e-11
M = 5.974e+24
m = 7.348e+22
R = 3.844e+8
omega = 2.662e-6

# Function
f = lambda r: (G*M)/(r**2.) - (G*m)/(R-r)**2. - omega**2*r
# Derivative
deriv = lambda r: -(2.*G*M)/(r**3.) - (2.*G*m)/(R-r)**3. - omega**2.

start = 2.*R/3.

# Newton's method
root = opt.root_scalar(f, method='newton', fprime=deriv, x0 = start)

print(root)
print(' ')

# Extracting items from object root, example:
print('The solution in units of R is: x/R = ', root.root/R )

"""Another syntax"""
root = opt.newton(f,start,deriv)
L1 = root
print('The solution in units of R is: x/R = ', root/R )

# Plotting
fig, ax = plt.subplots()
pos = np.arange(0.5 * R, 0.95 * R, 0.01 * R)
x = np.arange(0.5, 0.95, 0.01)
fx = f(pos)
ax.axhline(linestyle='dashed', color='red', linewidth=1)
ax.axvline(x=L1 / R, linestyle='dashed', color='red', linewidth=1)
ax.set_xlabel('x/R', size=16)
ax.set_ylabel('f(x)', size=16)
ax.annotate('$L_1$', xy=(2, 0.), xycoords='axes points', xytext=(270, -15), color='red', size=14)

ax.plot(x, fx)
plt.show()