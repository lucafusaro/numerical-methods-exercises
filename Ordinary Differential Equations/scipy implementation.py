import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define the differential equation
def model(y, t):
    dydt = -2 * y   #dy/dt = -2*y
    return dydt

# Set the initial condition
y0 = 1.0

# Create a time array
t = np.linspace(0, 5, 100)

# Solve the ODE using odeint
y = odeint(model, y0, t)

# Plot the solution
plt.plot(t, y)
plt.xlabel('Time')
plt.ylabel('y(t)')
plt.title('ODE Solution using odeint')
plt.show()


import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the differential equation
def model(t, y):
    dydt = -2 * y
    return dydt

# Set the initial condition
y0 = [1.0]

# Define the time span
t_span = (0, 5)

# Solve the ODE using solve_ivp
sol = solve_ivp(model, t_span, y0, t_eval=np.linspace(0, 5, 100))
# we can select the method with parameter method='RK45' or 'RK23' for example
# Plot the solution
plt.plot(sol.t, sol.y[0])
plt.xlabel('Time')
plt.ylabel('y(t)')
plt.title('ODE Solution using solve_ivp')
plt.show()

