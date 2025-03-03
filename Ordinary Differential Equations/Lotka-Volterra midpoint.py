"""The Lotka–Volterra equations, also known as the predator–prey equations,
are a pair of first-order nonlinear differential equations, frequently used to describe the dynamics
of biological systems in which two species interact, one as a predator and the other as prey.
The populations change through time according to the following system of equations:
dx/dt = α x − β x y,
dy/dt = γ x y − δ y,
where x is the number of preys and y the number of predators as a function of time t.
Write a python scripts that
A. integrates the system of equations with the midpoint scheme from time t = 0 to t = 50,
assuming x(0) = 10, y(0) = 10, α = 1.1, β = 0.4, γ = 0.1, δ = 0.4 [all of these quantities are
dimensionless]. You can use time-step h = 0.1;
B. plots the value of x(t) and y(t) as a function of time."""

import numpy as np
import matplotlib.pyplot as plt

specs = {'x0': [10., 10.], 'ti': 0., 'tf': 50., 'h': 0.1}


def velocity(x):
    alpha, beta, gamma, delta = 1.1, 0.4, 0.1, 0.4
    x_pos = x[0]
    y_pos = x[1]
    vx = alpha * x_pos - beta * x_pos * y_pos
    vy = gamma * x_pos * y_pos - delta * y_pos
    return np.array([vx, vy])


def midpoint(f, specs):
    h = specs['h']
    N = (specs['tf'] - specs['ti'])/h
    x_values = np.array(specs['x0'], dtype='float')
    Nequations = x_values.size
    x_t = np.zeros((int(N)+1, Nequations), dtype='float')
    x_t[0, :] = x_values
    t_values = np.arange(specs['ti'], specs['tf'], specs['h'])
    for i in range(int(N)):
        k1 = h * np.array(f(x_values))
        k2 = h * np.array(f(x_values + 0.5 * k1))
        x_values += k2
        x_t[i+1, :] = x_values
    return x_t[:-1], t_values


xt, t = midpoint(velocity, specs)

plt.plot(t, xt)
plt.scatter(t, xt[:, 0],s=5, label = 'preys', c = 'blue')
plt.scatter(t, xt[:, 1],s=5, label = 'predators', c = 'orange')
plt.xlabel('time t',size = 16)
plt.legend()
plt.show()