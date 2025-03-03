"""Consider the two-dimensional electronic capacitor shown in the Figure,
consisting of two flat metal plates (shown in blue in the Figure) enclosed in a square metal
box with 10 cm long side:
The plates are located as in the Figure and are 6 cm long. One of them is kept at a voltage
of +1 V and the other at −1 V. The walls of the box are at 0 V. Assuming the size of the plates
is negligible (i.e., it is the same as your numerical resolution),
A. write a script to solve the two-dimensional Laplace’s equation:
∂^2φ/∂x^2 + ∂^2φ/∂y2= 0
Suggestions: This is a static boundary problem. The method of finite differences is good for it.
A grid with 100 × 100 squared cells of side a = 0.1 cm will be perfect to solve this problem. A
tolerance of 10−3 is fine.
B. Plot the solution with a two-dimensional regular raster (i.e., use the matplotlib.pyplot.imshow()
function).
C. Try to solve this problem again with overrelaxation"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
Lx, Ly = 10.0, 10.0  # size of the box
Nx, Ny = 100, 100    # number of grid points
a = Lx / Nx          # grid spacing
# Initialize potential array
phi = np.zeros((Nx, Ny))

# Set boundary conditions
phi[0, :] = 0          # left boundary
phi[-1, :] = 0         # right boundary
phi[:, -1] = 0          # top boundary
phi[:, 0] = 0         # bottom boundary
phi[int(20*a), int(20*a) : int(80*a)] = 1  # +1 V plate, left plate
phi[int(80*a), int(20*a) : int(80*a)] = -1  # -1 V plate, right plate

# Tolerance for convergence
tolerance = 1e-3

# Iterative solution using the finite difference method
def difference_method(phi):
    while True:
        phi_old = phi.copy()
        for i in range(1, Nx-1):
            for j in range(1, Ny-1):
                phi[i, j] = 0.25 * (phi[i+1, j] + phi[i-1, j] + phi[i, j+1] + phi[i, j-1])

        if np.max(np.abs(phi - phi_old)) < tolerance:
            return phi


phi_diff = difference_method(phi)
# Plot the solution
plt.imshow(phi_diff.T, extent=[0, Lx, 0, Ly], origin='lower', cmap='viridis')
plt.colorbar(label='Potential (V)')
plt.title('Solution to Laplace\'s Equation')
plt.xlabel('x (cm)')
plt.ylabel('y (cm)')
plt.show()


# Iterative solution with overrelaxation
def overrelaxation(phi, omega):
    while True:
        phi_old = phi.copy()
        for i in range(1, Nx-1):
            for j in range(1, Ny-1):
                phi[i, j] = (1 - omega) * phi[i, j] + omega * 0.25 * (phi[i+1, j] + phi[i-1, j] + phi[i, j+1] + phi[i, j-1])

        if np.max(np.abs(phi - phi_old)) < tolerance:
            return phi

w = 1.8
phi_over = overrelaxation(phi, w)
# Plot the solution
plt.imshow(phi_over.T, extent=[0, Lx, 0, Ly], origin='lower', cmap='viridis')
plt.colorbar(label='Potential (V)')
plt.title('Solution to Laplace\'s Equation')
plt.xlabel('x (cm)')
plt.ylabel('y (cm)')
plt.show()
