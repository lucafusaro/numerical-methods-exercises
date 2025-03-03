"""Figure 1 describes a system of four springs with elastic constant k1, k2, k3 and k4.
The variables l1, l2, l3, and l4 are the relaxed lengths of each spring.
They are connected to three bodies of mass m1, m2, m3 as in the Figure. The three masses are initially
at distance x1(0), x2(0), and x3(0) from the origin of the (one-dimensional) rest frame. Assuming that the distance
between the walls is given by L = l1+l2+l3+l4, the motion of the three masses follows the system of three
second order ordinary differential equations (ODEs):
d^2x1/dt2 = −k1 (x1 − l1) + k2 (x2 − x1 − l2),
d^2x2/dt2 = −k2 (x2 − x1 − l2) + k3 (x3 − x2 − l3),
d^2x3/dt2 = −k3 (x3 − x2 − l3) + k4 (l1 + l2 + l3 − x3).
write a script that solves this system for l1 = l2 = l3 = l4 = 1 m, k1 = k2 = k3 = k4 = 1 N/m.
The initial values of the position are x1(0) = 0.7 m, x2(0) = 1.4 m, and x3(0) = 2.1 m. The
initial velocities are dx1/dt(0) = dx2/dt(0) = dx3/dt(0) = 0 m/s.
Write a python script that
A. integrates the system of equations with the Euler method from time t = 0 to t = 15 s
assuming time-steps h = 0.01 s;
B. plots the value of x1, x2 and x3 as a function of time t.
C. plots the value of dx1/dt, dx2/dt and dx3/dt as a function of time t"""

import numpy as np
import matplotlib.pyplot as plt

specs = {'x0': [0.7, 1.4, 2.1, 0., 0., 0.], 't0': 0., 't1': 15., 'h': 0.01}


def acceleration(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    l1, l2, l3, l4 = 1., 1., 1., 1.
    k1, k2, k3, k4 = 1., 1., 1., 1.
    a1 = -k1 * (x1 - l1) + k2 * (x2 - x1 - l2)
    a2 = -k2 * (x2 - x1 - l2) + k3 * (x3 - x2 - l3)
    a3 = -k3 * (x3 - x2 - l3) + k4 * (l1 + l2 + l3 - x3)
    return np.array([a1, a2, a3])


def euler(specs):

    x0 = specs['x0']
    h = specs['h']
    t0 = specs['t0']
    t1 = specs['t1']

    x1 = x0[0]
    x2 = x0[1]
    x3 = x0[2]

    x = np.array([x1, x2, x3])

    v1 = x0[3]
    v2 = x0[4]
    v3 = x0[5]

    v = np.array([v1, v2, v3])

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

        ynext = np.array([x[0], x[1], x[2], v[0], v[1], v[2]])
        y = np.vstack((y, ynext))
        times.append(t)
    return times, y


t, xt = euler(specs)
# Part B
x1 = xt[:, 0]
x2 = xt[:, 1]
x3 = xt[:, 2]
v1 = xt[:, 3]
v2 = xt[:, 4]
v3 = xt[:, 5]
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.scatter(t, x1, s=2, label = 'x1', c = 'blue')
plt.scatter(t, x2, s=2, label = 'x2', c = 'orange')
plt.scatter(t, x3, s=2, label = 'x3', c = 'green')
plt.title('Positions')
plt.xlabel('time')
plt.legend()
plt.subplot(2, 1, 2)
plt.scatter(t, v1, s=2, label = 'v1', c = 'blue')
plt.scatter(t, v2, s=2, label = 'v2', c = 'orange')
plt.scatter(t, v3, s=2, label = 'v3', c = 'green')
plt.title('Velocities')
plt.xlabel('time')
plt.tight_layout()
plt.show()