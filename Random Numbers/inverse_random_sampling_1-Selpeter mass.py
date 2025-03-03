"""use the inverse random sampling to generate the masses of
stars in a star cluster. Salpeter mass function: p(m) dm = const * m^-alpha dm,
with alpha = 2.3. Assuming that the minimum stellar mass is m_min= 0.1 Msun and
the maximum stellar mass is m_max = 150 Msun,
randomly calculate the mass of 10^6 stars distributed according to the Salpeter
initial mass function by using the inverse random sampling technique.
Plot the resulting population of stellar masses with an histogram
First calculate the const"""

import numpy as np
import matplotlib.pyplot as plt

m_min = 0.1 # 0.1 M_Sun
m_max = 250 #250 M_sun
alpha = 2.3
N =1e6
const = (1 - alpha) / (m_max ** (1 - alpha) - m_min ** (1 - alpha))

def salpeter_mass_generator(size):
    x = np.random.random(size)
    m_values = ((m_max ** (1 - alpha) - m_min ** (1 - alpha)) * x + m_min ** (1 - alpha)) ** (1/(1 - alpha))
    return m_values

"""if you want with a seed"""
#seed = secrets.randbits(128)
       # rng = np.random.default_rng(seed)
       # x = rng.random(self.size)

m_values = salpeter_mass_generator(int(N))

x = np.linspace(min(m_values), max(m_values), 100)
y = x ** (-alpha)

bin_edges = np.logspace(np.log10(min(m_values)), np.log10(max(m_values)), 30)
plt.hist(m_values, bins=bin_edges, density=True, histtype="step", color='blue', label='Generated Masses')
# Plot the reference line with angular coefficient -2/3
plt.plot(x, y, color='red', linestyle='--', label='y = m^(-2.3)')
plt.xlabel('Stellar Mass (Msun)')
plt.ylabel('PDF')
plt.title('Stellar Mass Distribution in a Star Cluster')
plt.loglog()
plt.legend()
plt.show()


