"""Consider three stars with masses  𝑚1=150, 𝑚2=200, 𝑚3=250,
   starting at rest in positions x1=(3,1), 𝑥2=(−1,−2), 𝑥3=(−1,1)
   Plot the orbits in the time interval [0,2], using an adaptive step integrator.
   Take 𝐺=1"""

import numpy as np
import matplotlib.pyplot as plt

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


class star:

    def __init__(self, m, x, y, vx, vy):
        self.mass = m
        self.position = np.array([x, y], dtype='float')
        self.velocity = np.array([vx, vy], dtype='float')

    def acceleration(self, star2):
        m2 = star2.mass
        r1 = self.position
        r2 = star2.position
        dist = r1 - r2
        r = np.sqrt(np.sum(dist ** 2))
        a1 = m2 * (r2 - r1) / (r ** 3)
        return a1


def f(x, t):
    # Input vector: x = [x1,y1,x2,y2,x3,y3,vx1,vy1,vx2,vy2,vx3,vy3]
    f = np.zeros(12)

    m1 = 150.
    m2 = 200.
    m3 = 250.

    star1 = star(m1, x[0], x[1], x[6], x[7])
    star2 = star(m2, x[2], x[3], x[8], x[9])
    star3 = star(m3, x[4], x[5], x[10], x[11])
    stars = [star1, star2, star3]

    # Output vector: f = [vx1,vy1,vx2,vy2,vx3,vy3,ax1,ay1,ax2,ay2,ax3,ay3]
    for i in range(6, 12):
        # print(i, x[i])
        f[i - 6] = x[i]

    for i in range(len(stars)):
        # print(i)
        a = np.zeros(2, dtype='float')
        s1 = stars[i]
        for s2 in stars[:i]:
            a += s1.acceleration(s2)
        for s2 in stars[i + 1:]:
            a += s1.acceleration(s2)
        f[2 * i + 6] = a[0]
        f[2 * i + 7] = a[1]
        # print(2*i+6,a[0])

    return f

specs = {'x0': [3.,1.,-1.,-2.,-1.,1.,0.,0.,0.,0.,0.,0.], 't0':0., 't1': 4., 'h0': 0.01, 'tol': 1e-4}

t, xt = ark4(f,specs)

a = 0
b = len(t)
plt.scatter(xt[a:b,0],xt[a:b,1],s=2, label = 'Star 1, $m_1 = 150$', c = 'blue')
plt.scatter(xt[a:b,2],xt[a:b,3],s=2, label = 'Star 2, $m_2 = 200$',c = 'darkorange')
plt.scatter(xt[a:b,4],xt[a:b,5],s=2, label = 'Star 3, $m_3 = 250$',c='green')
plt.scatter(xt[0,0],xt[0,1],s=50,c='blue')  #initial position
plt.scatter(xt[0,2],xt[0,3],s=50,c='darkorange')
plt.scatter(xt[0,4],xt[0,5],s=50,c='green')
#plt.plot(xt[a:b,0],xt[a:b,1], label = 'Star 1, $m_1 = 150$',c='blue')
#plt.plot(xt[a:b,2],xt[a:b,3], label = 'Star 2, $m_2 = 200$',c='darkorange')
#plt.plot(xt[a:b,4],xt[a:b,5], label = 'Star 3, $m_3 = 250$',c='green')
plt.xlabel('x',size=16)
plt.ylabel('y',size=16)
plt.legend()
plt.show()