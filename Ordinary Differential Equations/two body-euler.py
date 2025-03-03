"""Consider a system of two point masses i = 1, 2 subject to Newton gravity acceleration:
ai = −G * sum for j =! i of m_j * (xi - xj) / |xi - xj|^3
Assuming G = 1, the masses of the particles are m1 = 3 and m2 = 1, the initial positions are
x1 = (0, 0) and x2 = (1, 0) and the initial velocities are v1 = (0, 0) and v2 = (0, 2).
A. Write a script to integrate the above system between time t = 0 and time t = 5 with the
Euler method. Suggestion: use a fixed timestep h = 0.001.
B. Plot the time evolution of the positions of the two particles in the x, y plane, choosing the
center of mass as reference frame.
C. Calculate and plot the energy variation of the system [E(t) − E0]/E0, where E(t) is the
energy at time t for t ∈ [0, 5] and E0 = E(0) is the initial energy, and plot it as a function of
time.
D. Still using the Euler method, write a new script to integrate a system identical to the
previous one, but for which the acceleration is given by the Newton equation plus a drag term:
- beta * v_i where β = 0.1. Redo points B. and C. with the new script"""

import numpy as np
import matplotlib.pyplot as plt

specs = {'x0': [0., 0., 1., 0., 0., 0., 0., 2.], 't0': 0., 't1': 5., 'h': 0.001}
G = 1
m1 = 3.
m2 = 1.


def acceleration(x):
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


def euler(specs):

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

        k1x = h * v
        k1v = h * a

        x += k1x
        v += k1v

        t += h

        ynext = np.array([x[0], x[1], x[2], x[3], v[0], v[1], v[2], v[3]])
        y = np.vstack((y, ynext))
        times.append(t)
    return times, y


t, xt = euler(specs)
# Part B
a = 0
b = len(t)
X_center_of_mass = (m1 * xt[a:b, 0] + m2 * xt[a:b, 2]) / (m1 + m2)
Y_center_of_mass = (m1 * xt[a:b, 1] + m2 * xt[a:b, 3]) / (m1 + m2)
x1t_cm = xt[:, 0] - X_center_of_mass
x2t_cm = xt[:, 2] - X_center_of_mass
y1t_cm = xt[:, 1] - Y_center_of_mass
y2t_cm = xt[:, 3] - Y_center_of_mass
x1t_cm = x1t_cm.reshape(len(x1t_cm), 1)
x2t_cm = x2t_cm.reshape(len(x2t_cm), 1)
y1t_cm = y1t_cm.reshape(len(y1t_cm), 1)
y2t_cm = y2t_cm.reshape(len(y2t_cm), 1)
xt_cm = np.concatenate((x1t_cm, y1t_cm, x2t_cm, y2t_cm), axis=1)
plt.figure(figsize=(8, 6))
#plt.plot(X_center_of_mass, Y_center_of_mass, label='Center of Mass')
plt.scatter(xt_cm[a:b,0],xt_cm[a:b,1],s=2, label = 'Star 1, $m_1 = 3$', c = 'blue')
plt.scatter(xt_cm[a:b,2],xt_cm[a:b,3],s=2, label = 'Star 1, $m_1 = 1$', c = 'orange')
plt.title('Particle Trajectories with Euler Method (Without Drag Term)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

# Part C
# Calculate energy at each time step
kinetic_energy = 0.5 * (m1 * xt[:, 4] ** 2 + m1 * xt[:, 5] ** 2 +
                        m2 * xt[:, 6] ** 2 + m2 * xt[:, 7] ** 2)
potential_energy = -G * (m1 * m2) / np.sqrt((xt[:, 0] - xt[:, 2]) ** 2 +
                                            (xt[:, 1] - xt[:, 3]) ** 2)
total_energy = kinetic_energy + potential_energy

# Plot energy variation
plt.figure(figsize=(8, 6))
plt.plot(t, (total_energy - total_energy[0]) / np.abs(total_energy[0]), label='Energy Variation')
plt.title('Energy Variation with Euler Method (Without Drag Term)')
plt.xlabel('Time')
plt.ylabel('(E(t) - E(0)) / |E(0)|')
plt.legend()
plt.show()

# Part D
# Redo the simulation with the drag term
def acceleration_with_drag(x, v):
    beta = 0.1
    x1 = x[0]
    y1 = x[1]
    x2 = x[2]
    y2 = x[3]
    vx1 = v[0]
    vy1 = v[1]
    vx2 = v[2]
    vy2 = v[3]
    r = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    ax1 = m2 * (x2 - x1) / (r ** 3) - beta * vx1
    ay1 = m2 * (y2 - y1) / (r ** 3) - beta * vy1
    ax2 = m1 * (x1 - x2) / (r ** 3) - beta * vx2
    ay2 = m1 * (y1 - y2) / (r ** 3) - beta * vy2
    return np.array([ax1, ay1, ax2, ay2])

"""Rispetto alal funzione 'euler' cambia solo che dobbiamo passare ad a anche le velocità, potremmo 
quindi usare euler_drag anche per il caso senza drag effect"""
def euler_drag(specs):

    x0 = specs['x0']
    h = specs['h']
    t0 = specs['t0']
    t1 = specs['t1']

    x1 = x0[0]
    y1 = x0[1]
    x2 = x0[2]
    y2 = x0[3]

    vx1 = x0[4]
    vy1 = x0[5]
    vx2 = x0[6]
    vy2 = x0[7]

    x = np.array([x1, y1, x2, y2])
    v = np.array([vx1, vy1, vx2, vy2])

    times = [t0]
    y = np.copy(x0)

    t = t0

    while t < t1:
        a = acceleration_with_drag(x, v)  # a(t)

        k1x = h * v
        k1v = h * a

        x += k1x
        v += k1v

        t += h

        ynext = np.array([x[0], x[1], x[2], x[3], v[0], v[1], v[2], v[3]])
        y = np.vstack((y, ynext))
        times.append(t)
    return times, y   # di base cambia solo che a


t_drag, xt_drag = euler_drag(specs)
# Part B with drag eff
a = 0
b = len(t)
X_center_of_mass_d = (m1 * xt_drag[a:b, 0] + m2 * xt_drag[a:b, 2]) / (m1 + m2)
Y_center_of_mass_d = (m1 * xt_drag[a:b, 1] + m2 * xt_drag[a:b, 3]) / (m1 + m2)
x1t_cm_d = xt_drag[:, 0] - X_center_of_mass_d
x2t_cm_d = xt_drag[:, 2] - X_center_of_mass_d
y1t_cm_d = xt_drag[:, 1] - Y_center_of_mass_d
y2t_cm_d = xt_drag[:, 3] - Y_center_of_mass_d
x1t_cm_d = x1t_cm_d.reshape(len(x1t_cm_d), 1)
x2t_cm_d = x2t_cm_d.reshape(len(x2t_cm_d), 1)
y1t_cm_d = y1t_cm_d.reshape(len(y1t_cm_d), 1)
y2t_cm_d = y2t_cm_d.reshape(len(y2t_cm_d), 1)
xt_cm_d = np.concatenate((x1t_cm_d, y1t_cm_d, x2t_cm_d, y2t_cm_d), axis=1)
plt.figure(figsize=(8, 6))
#plt.plot(X_center_of_mass_d, Y_center_of_mass_d, label='Center of Mass')
plt.scatter(xt_cm_d[a:b,0],xt_cm_d[a:b,1],s=2, label = 'Star 1, $m_1 = 3$', c = 'blue')
plt.scatter(xt_cm_d[a:b,2],xt_cm_d[a:b,3],s=2, label = 'Star 1, $m_1 = 1$', c = 'orange')
plt.title('Particle Trajectories with Euler Method (With Drag Term)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

# Part C
# Calculate energy at each time step
kinetic_energy_d = 0.5 * (m1 * xt_drag[:, 4] ** 2 + m1 * xt_drag[:, 5] ** 2 +
                          m2 * xt_drag[:, 6] ** 2 + m2 * xt_drag[:, 7] ** 2)
potential_energy_d = -G * (m1 * m2) / np.sqrt((xt_drag[:, 0] - xt_drag[:, 2]) ** 2 +
                                              (xt_drag[:, 1] - xt_drag[:, 3]) ** 2)
total_energy_d = kinetic_energy_d + potential_energy_d

# Plot energy variation
plt.figure(figsize=(8, 6))
plt.plot(t, (total_energy_d - total_energy_d[0]) / np.abs(total_energy_d[0]), label='Energy Variation')
plt.title('Energy Variation with Euler Method (With Drag Term)')
plt.xlabel('Time')
plt.ylabel('(E(t) - E(0)) / |E(0)|')
plt.legend()
plt.show()
