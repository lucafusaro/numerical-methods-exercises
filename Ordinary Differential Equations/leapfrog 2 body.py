"""Assuming G = 1, the masses of the particles are m1 = 6. and m2 = 2., the initial positions are
and ~x1 = (0, 0, 0) and ~x2 = (0, 2., 0) and the initial velocities are ~v1 = (0, 0, 0) and ~v2 = (2., 0, 0).
A. Write a script to integrate the above system between time t = 0 and time t = 5 with the
leapfrog method. Suggestion: use a fixed timestep h = 0.01.
B. Plot the time evolution of the positions of the two particles in the x, y plane, choosing the
center of mass as reference frame.
C. Calculate and plot the energy variation of the system [E(t) − E0]/E0, where E(t) is the
energy at time t for t ∈ [0, 5] and E0 = E(0) is the initial energy, and plot it as a function of
time.
D. Calculate and plot the angular momentum variation of the system [J(t) − J0]/J0, where
J(t) is the magnitude of the angular momentum at time t for t ∈ [0, 5] and J0 = J(0) is the
initial angular momentum magnitude, and plot it as a function of time"""

import numpy as np
import matplotlib.pyplot as plt

specs = {'x0': [0., 0., 0., 0., 2., 0.], 'v0': [0., 0., 0., 2., 0., 0.], 't0': 0, 't1': 5., 'h': 0.01}

m1, m2 = 6., 2.
G = 1

def acc(x, t):
    x1, x2 = x[0], x[3]
    y1, y2 = x[1], x[4]
    z1, z2 = x[2], x[5]
    r = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
    ax1 = m2 * (x2 - x1) / (r ** 3)  # ax1 = dvx1/dt
    ay1 = m2 * (y2 - y1) / (r ** 3)  # ay1 = dvy1/dt
    az1 = m2 * (z2 - z1) / (r ** 3)
    ax2 = m1 * (x1 - x2) / (r ** 3)  # ax2 = dvx2/dt
    ay2 = m1 * (y1 - y2) / (r ** 3)  # ay2 = dvy2.dt
    az2 = m1 * (z1 - z2) / (r ** 3)
    return np.array([ax1, ay1, az1, ax2, ay2, az2])


def leapfrog(f, specs):
    x0 = np.array(specs['x0'])
    v0 = np.array(specs['v0'])
    t0 = specs['t0']
    t1 = specs['t1']
    h = specs['h']

    num_steps = int((t1 - t0) / h) + 1

    times = np.linspace(t0, t1, num_steps)
    positions = np.zeros((num_steps, len(x0)))
    velocities = np.zeros((num_steps, len(v0)))

    positions[0] = x0
    velocities[0] = v0

    for i in range(1, num_steps):
        # Update positions
        positions[i] = positions[i - 1] + h * velocities[i - 1] + 0.5 * h**2 * acc(positions[i - 1], times[i - 1])

        # Update accelerations at the new positions
        accelerations = acc(positions[i], times[i])

        # Update velocities
        velocities[i] = velocities[i - 1] + 0.5 * h * (acc(positions[i - 1], times[i - 1]) + accelerations)

    return times, positions, velocities


# A. Integrate the system using the leapfrog method
t, pos, vel = leapfrog(acc, specs)

# B. Plot the time evolution of the positions of the two particles in the x, y plane
X_center_of_mass = (m1 * pos[:, 0] + m2 * pos[:, 3]) / (m1 + m2)
Y_center_of_mass = (m1 * pos[:, 1] + m2 * pos[:, 4]) / (m1 + m2)
x1t_cm = pos[:, 0] - X_center_of_mass
x2t_cm = pos[:, 3] - X_center_of_mass
y1t_cm = pos[:, 1] - Y_center_of_mass
y2t_cm = pos[:, 4] - Y_center_of_mass
x1t_cm = x1t_cm.reshape(len(x1t_cm), 1)
x2t_cm = x2t_cm.reshape(len(x2t_cm), 1)
y1t_cm = y1t_cm.reshape(len(y1t_cm), 1)
y2t_cm = y2t_cm.reshape(len(y2t_cm), 1)
xt_cm = np.concatenate((x1t_cm, y1t_cm, x2t_cm, y2t_cm), axis=1)
plt.figure(figsize=(8, 6))
plt.scatter(xt_cm[:,0],xt_cm[:,1],s=2, label = '$m_1 = 6$', c = 'blue')
plt.scatter(xt_cm[:,2],xt_cm[:,3],s=2, label = '$m_2 = 2$', c = 'orange')
plt.title('Trajectories')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

# C. Calculate and plot the energy variation of the system
# Note: The energy variation is calculated as (E(t) - E0) / E0
# Calculate energy at each time step
kinetic_energy = 0.5 * (m1 * vel[:, 0] ** 2 + m1 * vel[:, 1] ** 2 + m1 * vel[:, 2] ** 2 +
                        m2 * vel[:, 3] ** 2 + m2 * vel[:, 4] ** 2 + m2 * vel[:, 5] ** 2)
potential_energy = -G * (m1 * m2) / np.sqrt((pos[:, 0] - pos[:, 3]) ** 2 +
                                            (pos[:, 1] - pos[:, 4]) ** 2 +
                                            (pos[:, 2] - pos[:, 5]) ** 2)
total_energy = kinetic_energy + potential_energy

# Plot energy variation
plt.figure(figsize=(8, 6))
plt.plot(t, (total_energy - total_energy[0]) / np.abs(total_energy[0]), label='Energy Variation')
plt.title('Energy Variation')
plt.xlabel('Time')
plt.ylabel('(E(t) - E(0)) / |E(0)|')
plt.legend()
plt.show()

# D Calculate and plot the angular momentum variation of the system
angular_momentum1 = m1 * np.cross(pos[:, :3], vel[:, :3])
angular_momentum2 = m2 * np.cross(pos[:, 3:], vel[:, 3:])
ang_mom = angular_momentum1 + angular_momentum2
angular_momentum_magnitude = np.linalg.norm(ang_mom, axis=1)

plt.figure(figsize=(12, 6))
plt.plot(t, (angular_momentum_magnitude - angular_momentum_magnitude[0]) / angular_momentum_magnitude[0])
plt.xlabel('Time')
plt.ylabel('(J(t) - J0) / J0')
plt.title('Angular Momentum Variation of the System')
plt.show()


