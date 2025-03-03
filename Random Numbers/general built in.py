import numpy as np
import secrets # This contains functions to generate large random seeds

"""Generation of uniform r.v.'s"""
# Set up initial seed
seed = secrets.randbits(128)   # Generate large positive integer: best kind of seed for the generator
print('seed = ', seed, '\n')

# Set up object rng = uniform random number generator (with default numpy choice = PCG-64 generator)
rng = np.random.default_rng(seed)

# Now we simply use the function 'random', applied to the object rng, to generate uniform distributed r.v.'s
# We can arrange them in matrices with desired shape.

# Just one variable
x = rng.random()
print('x = ', x, '\n')

# 100 r.v.'s arranged in a vector
rvs = rng.random(100)
print('Vector = ', rvs, '\n')

# 100 r.v.'s arranged in a 10x10 matrix
rvs2 = rng.random( (10,10) )
print('Matrix = ', rvs2, '\n')

"""Generation of gaussian r.v.'s"""
mean = 0.
std = 1.5

seed = 1243487
np.random.seed(seed)

# Just one variable
x = np.random.normal(mean,std)
print('x = ', x, '\n')

# 100 r.v.'s arranged in a vector
rvs = np.random.normal(mean,std,size=100)
print('Vector = ', rvs, '\n')

# 100 r.v.'s arranged in a 10x10 matrix
rvs2 = np.random.normal(mean,std,size=(10,10) )
print('Matrix = ', rvs2, '\n')


# Generating Gaussian r.v.'s and plotting histogram

import numpy as np
import secrets
import matplotlib.pyplot as plt
from scipy.stats import norm

# Mean and standard deviation
mean = 3.5
std = 8.2
# Size of sample
N = 100000

# Seed
seed = secrets.randbits(32)
np.random.seed(seed)
#print(seed)
# Generating data
rvs = np.random.normal(mean,std,size=N)

# Generating pdf to overplot
xmin = mean - 5.*std
xmax = mean  + 5.*std
step = 0.1*std
x_axis = np.arange(xmin,xmax,step)
pdf = norm.pdf(x_axis,mean,std)

# Data histogram
plt.hist(rvs, bins=100, density=True, ec = 'black', histtype='bar')
# pdf plot
plt.plot(x_axis,pdf,label='$\mu$ = ' + str(mean) + ',' + ' $\sigma$ = ' + str(std))
plt.legend()
plt.xlabel('x',size = 18)
plt.ylabel('$N(\mu,\sigma)$', size = 18)
plt.show()

seed = secrets.randbits(128)   # Generate large positive integer: best kind of seed for the generator
#print('seed = ', seed, '\n')

# Set up object rng = Gaussian random number generator
# (with default numpy choice for uniform r.v.'s generator'= PCG-64 generator)
mean = 3.5
std = 8.2
N = 100000

# Generate Gaussian r.v's
rvs = np.random.default_rng(seed).normal(mean,std,N)

# Generating pdf to overplot
xmin = mean - 5.*std
xmax = mean + 5.*std
step = 0.1*std
x_axis = np.arange(xmin,xmax,step)
pdf = norm.pdf(x_axis,mean,std)

# Data histogram
plt.hist(rvs, bins=100, density=True, ec = 'black', histtype='bar')
# pdf plot
plt.plot(x_axis,pdf,label='$\mu$ = ' + str(mean) + ',' + ' $\sigma$ = ' + str(std))
plt.legend()
plt.xlabel('x',size = 18)
plt.ylabel('$N(\mu,\sigma)$', size = 18)
plt.show()
