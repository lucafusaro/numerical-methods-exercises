"""The following system of two second-order ordinary differential equations
(ODEs) describes the motion of a two-dimensional harmonic oscillator:
d^2x/dt^2 = −ω^2 x
d^2y/dt^2 = −ω^2 y, with constant ω = 1.0 s^−1
At time t = 0, the positions are (x, y) = (2, 0) m, and the velocities
are (vx, vy) = (0, 2) m s^−1
Write a python script that
A. Integrates the evolution of the system with the Euler scheme from time t0 = 0 to tf = 10 s
with a time-step h = 0.01.
B. Plots the trajectory of the oscillator in the x, y plane.
The damped two-dimensional harmonic oscillator is defined by this other system of second order ODEs:
d^2x/dt^2 = −ω^2 x - 2 ζ ω v_x
d^2y/dt^2 = −ω^2 y − 2 ζ ω v_y
where ω is the same as above, while ζ = 0.1. The initial positions and velocities are the same
as above.
Write a new script (or a new part of your previous script) that
C. Integrates the evolution of the system with the Euler scheme from time t0 = 0 to tf = 10 s
with a time-step h = 0.01.
D. Plots the trajectory of the oscillator in the x, y plane."""

import numpy as np
import matplotlib.pyplot as plt

specs = {'x0': [2., 0., 0., 2.,], 't0': 0., 't1': 10., 'h': 0.01}

def acceleration(x):
    w = 1.0  #s^−1
    x_pos = x[0]
    y_pos = x[1]
    ax1 = -w**2 * x_pos  # ax1 = dvx1/dt
    ay1 = -w**2 * y_pos  # ay1 = dvy1/dt
    return np.array([ax1, ay1])


def euler(specs):

    x0 = specs['x0']
    h = specs['h']
    t0 = specs['t0']
    t1 = specs['t1']

    x_pos = x0[0]
    y_pos = x0[1]

    x = np.array([x_pos, y_pos])

    vx = x0[2]
    vy = x0[3]

    v = np.array([vx, vy])

    times = [t0]
    y = np.copy(x0)

    t = t0

    while t < t1:
        a = acceleration(x)  # a(t)

        k1x = h * v
        k1v = h * a

        x += k1x
        v += k1v

        t += h

        ynext = np.array([x[0], x[1], v[0], v[1]])
        y = np.vstack((y, ynext))
        times.append(t)
    return times, y


t, xt = euler(specs)
# Part B
a = 0
b = len(t)
plt.figure(figsize=(8, 6))
plt.scatter(xt[a:b, 0], xt[a:b, 1], s=2, c='blue')
plt.title('Particle Trajectory with Euler Method (Without Damp Term)')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

#part C
# Redo the simulation with the damp term
def acceleration_with_damp(x, v):
    w = 0.1
    zeta = 0.1
    x_pos = x[0]
    y_pos = x[1]
    vx = v[0]
    vy = v[1]
    ax = -w**2 * x_pos - 2 * zeta * w * vx
    ay = -w**2 * y_pos - 2 * zeta * w * vy
    return np.array([ax, ay])

"""Rispetto alal funzione 'euler' cambia solo che dobbiamo passare ad a anche le velocità, potremmo 
quindi usare euler_drag anche per il caso senza drag effect"""
def euler_damp(specs):

    x0 = specs['x0']
    h = specs['h']
    t0 = specs['t0']
    t1 = specs['t1']

    x_pos = x0[0]
    y_pos = x0[1]

    x = np.array([x_pos, y_pos])

    vx = x0[2]
    vy = x0[3]

    v = np.array([vx, vy])

    times = [t0]
    y = np.copy(x0)

    t = t0

    while t < t1:
        a = acceleration_with_damp(x, v)  # a(t)

        k1x = h * v
        k1v = h * a

        x += k1x
        v += k1v

        t += h

        ynext = np.array([x[0], x[1], v[0], v[1]])
        y = np.vstack((y, ynext))
        times.append(t)
    return times, y



t_damp, xt_damp = euler_damp(specs)
# Part B with drag eff
a = 0
b = len(t)
plt.figure(figsize=(8, 6))
plt.scatter(xt_damp[a:b, 0], xt_damp[a:b, 1], s=2, c='blue')
plt.title('Particle Trajectory with Euler Method (With Damp Term)')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
