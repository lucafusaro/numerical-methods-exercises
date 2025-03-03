"""Write a script to generate N = 10^5 Gaussian deviates with the Box - Muller method.
Assume sigma = 2 and that the gaussian is centered on zero"""

import random
import numpy as np
import matplotlib.pyplot as plt
N = 1e5
sigma = 2
def box_muller(z1, z2):
    r = np.sqrt(-2 * (sigma**2) * np.log(1 - z1))
    theta = 2 * np.pi * z2
    x = r * np.cos(theta)   #if we want a different mean value we can shift, tipo x = r cos(theta) -2 è centrata su -2
    y = r * np.sin(theta)
    return x, y

def gaussian(x):
    y = (1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(1/2) * (x / sigma)**2))
    return y


x_values = []
y_values = []
gaussian_values = []


z1 = np.random.random(int(N))
z2 = np.random.random(int(N))
x = box_muller(z1, z2)[0]
y = box_muller(z1, z2)[1]
# note you obtain two gaussian distributed variable x and y
plt.hist(x, density=True, bins=100)
x_val = np.arange(-7.5, 7.5, 0.1)
gauss_val = np.vectorize(gaussian)
plt.plot(x_val, gauss_val(x_val))
plt.xlabel("x")
plt.ylabel("PDF(x)")
plt.show()

"""Con funzioni già esistenti in python:"""
sample1 = np.random.normal(loc=0.0, scale=2.0, size=(2,2)) #genera un valore estratto da pdf gaussiana con mean value = loc, std dev = scale
sample2 = random.gauss(0.0, 2.0) #primo argomento mean value, secondo std dev