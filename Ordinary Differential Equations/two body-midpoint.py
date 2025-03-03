"""Solve the two body problem. Take G=1; set masses  𝑚1=𝑚2=1. Initial positions
  x1=(1,0), x2=(−1,0). Initial velocities  v1=(−0.3,0.3), v2=(0.3,−0.3).
  Implement the midpoint method"""

import numpy as np
import matplotlib.pyplot as plt

specs = {'x0': [1., 0., -1., 0., -0.3, 0.3, 0.3, -0.3], 't0': 0., 't1': 12., 'h': 0.001}


def midpoint_2body(specs):
    def acceleration(x):
        m1 = 1.
        m2 = 1.
        x1 = x[0]
        y1 = x[1]
        x2 = x[2]
        y2 = x[3]
        r = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        ax1 = m2 * (x2 - x1) / (r ** 3)  # ax1 = dvx1/dt
        ay1 = m2 * (y2 - y1) / (r ** 3)  # ay1 = dvy1/dt
        ax2 = m1 * (x1 - x2) / (r ** 3)  # ax2 = dvx2/dt
        ay2 = m1 * (y1 - y2) / (r ** 3)  # ay2 = dvy2.dt
        return np.array([ax1, ay1, ax2, ay2])

    x0 = specs['x0']
    h = specs['h']
    t0 = specs['t0']
    t1 = specs['t1']

    x1 = x0[0]
    y1 = x0[1]
    x2 = x0[2]
    y2 = x0[3]

    x = np.array([x1, y1, x2, y2])

    vx1 = x0[4]
    vy1 = x0[5]
    vx2 = x0[6]
    vy2 = x0[7]

    v = np.array([vx1, vy1, vx2, vy2])

    times = [t0]
    y = np.copy(x0)

    t = t0

    while t < t1:
        a = acceleration(x)  # a(t)

        k1x = 0.5 * h * v
        k1v = 0.5 * h * a

        x1 = x + k1x  # x(t + h/2)

        a1 = acceleration(x1)  # a(t+h/2)

        k2x = h * (v + k1v)
        k2v = h * a1

        x += k2x
        v += k2v

        t += h

        ynext = np.array([x[0], x[1], x[2], x[3], v[0], v[1], v[2], v[3]])
        y = np.vstack((y, ynext))
        times.append(t)

    return times, y


t, xt = midpoint_2body(specs)

a = 0
b = len(t)
plt.scatter(xt[a:b,0],xt[a:b,1],s=2, label = 'Star 1, $m_1 = 1$', c = 'blue')
plt.scatter(xt[a:b,2],xt[a:b,3],s=2, label = 'Star 1, $m_1 = 1$', c = 'orange')
plt.xlabel('x',size = 16)
plt.ylabel('y',size=16)
plt.show()