"""Use rk4 or adaptive rk4 to calculate the orbit of a comet with distance
   from sun at aphelion 𝑑=4×10^9 km and velocity at aphelion  v=500 m/s"""

import numpy as np
import matplotlib.pyplot as plt

def f(x,t):
    G = 6.6743e-11  # Newton's constant m^3 Kg^-1 s^-2
    sec2years = 3.17098e-8
    m2km = 1e-3
    G = (G * m2km ** 3) / (sec2years ** 2)  # converting sec to years and meters to km
    M = 1.98847e30  # Sun's mass Kg
    GM = G * M

    x1 = x[0]  # x
    v1 = x[1]  # vx
    x2 = x[2]  # y
    v2 = x[3]  # vy

    # Comet distance from the sun
    r = np.sqrt(x1 ** 2. + x2 ** 2.)
    # Array of derivatives
    f0 = v1  # dx/dt = v
    # we're considering sun at origin
    f1 = -GM * x1 / r ** 3  # dvx/dt -GM*x/r^3
    f2 = v2  # dy/dt = vy
    f3 = -GM * x2 / r ** 3  # dvy/dt = -GM*y/r^3

    return np.array([f0, f1, f2, f3])

# Setting initial conditions. Comet at aphelion
#################################################
# Calculating G*M in chosen units (distances in km and time in years)
G = 6.6743e-11  # Newton's constant
sec2years = 3.17098e-8
m2km = 1e-3
G = (G*m2km**3)/(sec2years**2) # converting sec to years and meters to km
M = 1.98847e30  # Sun's mass
GM = G*M

# Initial position (xi,yi), distances in km
xi  = 4e9    # note that at aphelion (or perielion) considering Sun at origin, distance is fully on x
yi  = 0.
# Initial velocity (vx,vy), in km/yr
vx0 = 0.       # obviously all the velocity at the aphelion (or perielion) is on y (ellipse orbit)
sec2years = 3.17098e-8
# Velocity at aphelion
vy0 = 0.5/sec2years
# Major semi-axis (in km), derived from velocity and distance at aphelion
a = (2./xi - vy0**2/GM)**(-1)
# Calculating period of orbit (in years), from Kepler's third law
T = 2.*np.pi*np.sqrt((a**3)/GM)
print("T = ", T)

# Setting specs using results above. Setting tolerance in km/yr on comet position. Step in km.
# Integrating from 0 to T
specs = {'x0': [xi, 0., 0., vy0], 'ti':0, 'tf': T, 'h': 1e-3}
#x0 : x, vx, y, vy


def RK4(f, specs):
    h = specs['h']
    N = (specs['tf'] - specs['ti'])/h
    x_values = np.array(specs['x0'], dtype='float')
    Nequations = x_values.size
    x_t = np.zeros((int(N)+1, Nequations), dtype='float')
    x_t[0, :] = x_values
    t_values = np.arange(specs['ti'], specs['tf'], h)
    for i in range(int(N)):
        t = t_values[i]
        k1 = h * np.array(f(x_values, t))
        k2 = h * np.array(f(x_values + 0.5 * k1, t + 0.5 * h))
        k3 = h * np.array(f(x_values + 0.5 * k2, t + 0.5 * h))
        k4 = h * np.array(f(x_values + k3, t + h))
        x_values += (1./6.)*(k1 + 2.*k2 + 2.*k3 + k4)
        x_t[i+1, :] = x_values
    return x_t[:-1], t_values


xt, t = RK4(f, specs)
#a = 0
#b = len(xt)
plt.scatter(xt[:,0],xt[:,2],s=5, color='red')
#plt.scatter(t[a:b], xt[:, 1])
#plt.scatter(t[a:b], xt[:, 3])
plt.xlabel('x(t)',size=16)
plt.ylabel('y(t)',size=16)
plt.title('Comet orbit',size=16)
plt.show()