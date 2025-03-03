"""Use the Verlet method to calculate the orbit of the Earth around the sun.
   Verify energy conservation"""

import numpy as np
import matplotlib.pyplot as plt


def f(x, t):
    G = 6.6743e-11  # Newton's constant
    sec2years = 3.17098e-8
    m2km = 1e-3
    G = (G * m2km ** 3) / (sec2years ** 2)  # converting sec to years and meters to km
    M = 1.98847e30  # Sun's mass
    GM = G * M

    x1 = x[0]  # x
    x2 = x[1]  # y

    r = np.sqrt(x1 ** 2. + x2 ** 2.)

    f0 = -GM * x1 / r ** 3  # dvx/dt = -GM*x/r^3
    f1 = -GM * x2 / r ** 3  # dvy/dt = -GM*y/r^3

    return np.array([f0, f1])


def verlet(f, specs):
    x0 = np.array(specs['x0'])
    v0 = np.array(specs['v0'])
    t0 = specs['t0']
    t1 = specs['t1']
    h = specs['h']

    # Solution
    xt = np.copy(x0)
    vt = np.copy(v0)
    # Times at which the solution is computed
    times = [t0]
    t = t0

    # Energy
    G = 6.6743e-11  # Newton's constant
    sec2years = 3.17098e-8
    m2km = 1e-3
    G = (G * m2km ** 3) / (sec2years ** 2)  # converting sec to years and meters to km
    M = 1.98847e30  # Sun's mass
    m = 5.9722e24  # Earth's mass
    GMm = G * M * m
    r = np.sqrt(np.sum(x0 ** 2))
    Ep = -GMm / r
    Ek = 0.5 * m * np.sum(v0 ** 2)
    Etot = [Ep + Ek]

    v1 = v0 + 0.5 * h * f(x0, t0)  # v(t+h/2)
    x = np.copy(x0)
    x += h * v1

    while t < t1:
        t += h
        x += h * v1  # x(t+h) = x(t) + h*v1
        k = h * f(x, t)  # k = h*f(x(t+h),t+h)
        v = v1 + 0.5 * k  # v(t+h) = v(t+h/2) + k/2
        v1 += k  # v(t + 3h/2) = v(t+h/2) + k
        times.append(t)
        xt = np.vstack((xt, x))
        vt = np.vstack((vt, v))
        # Energy
        r = np.sqrt(np.sum(x ** 2))
        Ep = -GMm / r
        Ek = 0.5 * m * np.sum(v ** 2)
        Etot.append(Ep + Ek)

    return times, xt, vt, Etot


xi = 1.471e8  # Distance at perihelion in km
sec2years = 3.17098e-8
vy0 = -3.0287e1 / sec2years  # Velocity at perihelion, km/yr

specs = {'x0': [xi, 0.], 'v0': [0., vy0], 't0': 0, 't1': 10., 'h': 1e-4}

t, xt, vt, Etot = verlet(f, specs)
plt.scatter(t, vt[:, 0])
plt.scatter(t, vt[:, 1])
#plt.scatter(xt[:,0],xt[:,1],s=5, color='red')
plt.xlabel('x(t)',size=16)
plt.ylabel('y(t)',size=16)
plt.title('Earth orbit',size=16)
plt.show()