import numpy as np
import matplotlib.pyplot as plt

# Define the system of ODEs
def f(t, u):
    x, y = u
    dxdt = -y
    dydt = x
    return np.array([dxdt, dydt])

# Leapfrog method implementation for a system of ODEs
def leapfrog_system(h, t, u0, f):
    n_steps = len(t)
    u = np.zeros((n_steps, len(u0)))

    u[0] = u0

    for i in range(1, n_steps):
        # Half-step for velocities
        u[i - 1] += 0.5 * h * f(t[i - 1], u[i - 1])

        # Full step for positions
        u[i] = u[i - 1] + h * u[i - 1]

        # Full step for velocities
        u[i] += 0.5 * h * f(t[i], u[i])

    return u

# Parameters
t_start = 0.0
t_end = 10.0
h = 0.1  # Time step
u0 = np.array([1.0, 0.0])  # Initial conditions

# Generate time array
t = np.arange(t_start, t_end + h, h)

# Solve the system of ODEs using leapfrog method
solution = leapfrog_system(h, t, u0, f)

# Plot the results
plt.plot(t, solution[:, 0], label='x(t)')
plt.plot(t, solution[:, 1], label='y(t)')
plt.xlabel('Time')
plt.ylabel('Variables')
plt.title('Solution of the System of ODEs using Leapfrog Method')
plt.legend()
plt.show()
