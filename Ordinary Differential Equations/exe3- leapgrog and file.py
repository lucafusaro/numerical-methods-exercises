"""Download the file five bodies.txt. It contains the information about the
position, velocity and mass of five bodies subject to gravitational force, with acceleration:
ai = −G sum for j=!i,j≤5 of mj (xi − xj)||xi − xj|^3
where i is the index of the body, j = 1, 2, 3, 4, 5 is an index that runs over the five bodies, G is
the gravity constant, mj is the mass of the jth body, xi is the position vector of the ith body
and vi is the velocity vector of the ith body.
In the aforementioned file, columns 0, 1 and 2 are the x, y and z components of the initial
position (in astronomical units, AU); columns 3, 4 and 5 are the x, y and z components of the
initial velocity (in astronomical units per year, AU/yr); column 6 is the mass in Earth masses
(MEarth). Write a python script that addresses the following points.
A. Integrate the evolution of the five bodies with a leapfrog scheme between time t = 0 and
t = 3 yr, with steps of 10−2 yr.
B. Plot the time evolution of the x, y components.
C. Calculate and plot the total energy as a function of time.
D. Calculate and plot the magnitude of the angular momentum as a function of time.
E. Can you figure out if the five bodies resemble some astrophysical system you know?
Suggestion: Remember to set the right units for the gravity constant G"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 4 * np.pi**2  # Gravitational constant in AU^3 / (M_Earth yr^2)
AU_to_m = 1.496e11  # Conversion factor from astronomical units to meters
Earth_mass_to_kg = 5.972e24  # Conversion factor from Earth masses to kilograms
yr_to_s = 365.25 * 24 * 60 * 60  # Conversion factor from years to seconds

# Load data from the file
data = np.loadtxt('five_bodies.txt')

# Extract relevant information from the data
positions_initial = data[:, :3]  # Initial positions in astronomical units
velocities_initial = data[:, 3:6]  # Initial velocities in AU/yr
masses = data[:, 6]  # Masses in Earth masses

# Function to compute gravitational accelerations
def gravitational_accelerations(positions):
    num_bodies = len(positions)
    accelerations = np.zeros_like(positions)

    for i in range(num_bodies):
        for j in range(num_bodies):
            if i != j:
                r = positions[j] - positions[i]
                accelerations[i] += G * masses[j] * r / np.linalg.norm(r)**3

    return accelerations

# Leapfrog integration
def leapfrog_integration(positions, velocities, dt, num_steps):
    times = np.zeros(num_steps)
    all_positions = np.zeros((num_steps, len(positions), 3))
    all_velocities = np.zeros((num_steps, len(velocities), 3))

    for step in range(num_steps):
        times[step] = step * dt

        # Update velocities
        accelerations = gravitational_accelerations(positions)
        velocities += 0.5 * dt * accelerations

        # Update positions
        positions += dt * velocities

        # Update velocities again
        accelerations = gravitational_accelerations(positions)
        velocities += 0.5 * dt * accelerations

        # Store positions and velocities
        all_positions[step] = positions
        all_velocities[step] = velocities

    return times, all_positions, all_velocities

# Time parameters
t_initial = 0.0
t_final = 3.0
dt = 1e-2
num_steps = int((t_final - t_initial) / dt)

# Perform leapfrog integration
times, all_positions, all_velocities = leapfrog_integration(positions_initial, velocities_initial, dt, num_steps)

# Plotting the time evolution of x and y components in two different panels
plt.figure(figsize=(12, 8))

# Panel for x positions
plt.subplot(2, 1, 1)
for i in range(5):
    plt.plot(times, all_positions[:, i, 0], label=f'Body {i+1} - x')

plt.xlabel('Time (years)')
plt.ylabel('Position (AU)')
plt.title('Time Evolution of x Components')
plt.legend()

# Panel for y positions
plt.subplot(2, 1, 2)
for i in range(5):
    plt.plot(times, all_positions[:, i, 1], label=f'Body {i+1} - y')

plt.xlabel('Time (years)')
plt.ylabel('Position (AU)')
plt.title('Time Evolution of y Components')
plt.legend()

plt.tight_layout()
plt.show()

# ... (previous code remains unchanged)

# Function to calculate total energy
def calculate_total_energy(positions, velocities, masses):
    kinetic_energy = 0.5 * np.sum(masses * np.linalg.norm(velocities, axis=2)**2)
    gravitational_energy = 0.0

    num_bodies = len(positions[0])
    for i in range(num_bodies):
        for j in range(i+1, num_bodies):
            r = positions[:, j] - positions[:, i]
            gravitational_energy -= G * masses[i] * masses[j] / np.linalg.norm(r, axis=1)

    return kinetic_energy + gravitational_energy

# Calculate total energy as a function of time
total_energy = calculate_total_energy(all_positions, all_velocities, masses)

# Plot total energy
plt.figure(figsize=(12, 6))
plt.plot(times, total_energy)
plt.xlabel('Time (years)')
plt.ylabel('Total Energy (Joules)')
plt.title('Total Energy as a Function of Time')
plt.show()

# Function to calculate magnitude of angular momentum
def calculate_angular_momentum(positions, velocities, masses):
    angular_momentum = np.cross(positions, masses[:, None] * velocities)
    return np.linalg.norm(angular_momentum, axis=2)

# Calculate angular momentum as a function of time
angular_momentum_magnitude = calculate_angular_momentum(all_positions, all_velocities, masses)
angular_momentum_total = np.sum(angular_momentum_magnitude, axis=1)

# Plot magnitude of angular momentum
plt.figure(figsize=(12, 6))
plt.plot(times, angular_momentum_total)
plt.xlabel('Time (years)')
plt.ylabel('Angular Momentum Magnitude (kg m^2/s)')
plt.title('Magnitude of Angular Momentum as a Function of Time')
plt.show()


