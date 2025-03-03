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
specs = {'x0': [xi, 0., 0., vy0], 't0':0, 't1': T, 'h0': 1., 'tol': 1.}
#x0 : x, vx, y, vy
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


t, xt = ark4(f, specs)

plt.scatter(xt[:,0],xt[:,2],s=5, color='red')
plt.xlabel('x(t)',size=16)
plt.ylabel('y(t)',size=16)
plt.title('Comet orbit',size=16)
plt.show()