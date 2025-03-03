"""Use your rk4 or adaptive rk4 to solve the Lotka-Volterra system,
 describing the prey (y = "foxes") - predator (x = "rabbits") population dynamics:
 dx/dt = ax - 𝛽xy, dy/dt = 𝛾xy − 𝛿y
with 𝛼=1, 𝛽=𝛾=0.5, 𝛿=2. Evolve the system from  t0=0 to  t1=30,
starting from initial conditions  𝑥0=𝑦0=2"""

import numpy as np
import matplotlib.pyplot as plt


def f(x,t):
    x1 = x[0]
    x2 = x[1]
    alpha, beta, gamma, delta = 1.0, 0.5, 0.5, 2
    f1 = alpha*x1 - beta*x1*x2
    f2 = gamma*x1*x2 - delta*x2
    return [f1,f2]

def ark4(f, specs):
    # Solve ODE with 4th order Runge-Kutta
    ####
    def increment(f, x, t, h):
        k1 = h * np.array(f(x, t))
        k2 = h * np.array(f(x + 0.5 * k1, t + 0.5 * h))
        k3 = h * np.array(f(x + 0.5 * k2, t + 0.5 * h))
        k4 = h * np.array(f(x + k3, t + h))
        deltax = (1. / 6.) * (k1 + 2. * k2 + 2. * k3 + k4)
        return deltax

    # Setting up starting step and initial conditions
    h = (specs['h0'])
    x = np.array(specs['x0'], dtype='float')
    # Array in which the solution will be stored.
    # Start with one row and append rows at each succesful
    # iteration. The number of columns is equal to the
    # number of equations in the system
    xt = np.copy(x)
    # Array containing times at which the solution is computed
    t = specs['t0']
    ts = []
    ts.append(t)
    while t < specs['t1']:
        # Find x1, increment twice by h
        dx1 = increment(f, x, t, h)
        x1 = x + dx1
        dx1 = increment(f, x1, t + h, h)
        x1 = x1 + dx1
        # Find x2, increment once by 2h
        dx2 = increment(f, x, t, 2 * h)
        x2 = x + dx2
        # Estimate error
        eps = (abs(x2 - x1)) / 30.
        err = np.sqrt(np.sum(eps ** 2))
        # Compare error to target tolerance
        if (err <= 1e-60):
            # Avoid overflow in rho
            rho = 1.0
        else:
            rho = (specs['tol'] * h) / err
        # Accepting or rejceting step, depending on whether
        # the error is below or above target.
        if (rho >= 1):
            # x1 is accepted
            # Update time
            t += 2 * h
            # Update x
            x = x1
            # print(t,x)
            # Stacking x as a new row to array of solutions
            xt = np.vstack([xt, x])
            # Appending time step
            ts.append(t)
            # Updating step
            h = min(h * (rho ** 0.25), 2. * h)
        elif (rho < 1.):
            # Not enough accuracy, stay in x
            h = min(h * (rho ** 0.25), 0.99999 * h)

    return np.array(ts), xt

specs = {'x0': [2. ,2.], 't0':0., 't1':30., 'h0': 0.1, 'tol': 1e-5}


t, xt = ark4(f,specs)

plt.plot(t,xt)
plt.scatter(t,xt[:,0],s = 10, label='x(t):Rabbits')
plt.scatter(t,xt[:,1], s = 10, label='y(t):Foxes')
plt.legend(fontsize=12)
plt.xlabel('t',size=16)
plt.show()