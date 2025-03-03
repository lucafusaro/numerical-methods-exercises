"""Write a script to generate N = 10^5 random numbers following a
Maxwellian distribution with sigma=265 km/s. According to Hobbs et al. [2005], this
is the distribution of natal kicks of neutron stars.
Note: every compact object which forms from a supernova is thought to
receive a kick at birth. The main reason is that linear momentum is
conserved during a supernova and asymmetries in the ejecta (or in neutrino losses)
push the compact object to move in the opposite direction with respect to the bulk of the ejecta."""

import numpy as np
import matplotlib.pyplot as plt
def maxwellian(v, sigma):
    return np.sqrt(2 / np.pi) * (v**2 / sigma**3) * np.exp(-v**2 / (2 * sigma**2))

def v(x, y, z):
    return np.sqrt(x**2 + y**2 + z**2)


N = 1e5
maxw_sigma = 265
maxw_values = []
for i in range(int(N)):
    x = np.random.normal(loc=0.0, scale=maxw_sigma)
    y = np.random.normal(loc=0.0, scale=maxw_sigma)
    z = np.random.normal(loc=0.0, scale=maxw_sigma)
    maxw_values.append(v(x, y, z))

"""Plotting"""
plt.hist(maxw_values, bins=100, density=True, label="Generated points")
v_values = np.linspace(0, max(maxw_values), 1000)
plt.plot(v_values, maxwellian(v_values, maxw_sigma), label="Mawxwellian pdf")
plt.xlabel("3D-speed v (km/s)")
plt.ylabel("Maxwelliand pdf")
plt.legend()
plt.show()
