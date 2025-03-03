"""Write a script that
A. generates 10^4 data points randomly distributed inside a homogeneous and isotropic
sphere of radius R = 5;
B. plots the result with a three-dimensional scatter plot, in which the axes indicate the x, y
and z coordinates.
Suggestion: you can use either the inverse random sampling (starting from spherical coordinates)
or the rejection method (starting from Cartesian coordinates)."""

import numpy as np
import matplotlib.pyplot as plt
import secrets

"""Inverse random sampling method"""
def pol2cart(r, theta, phi):
    # Converting polar into Cartesian coordinates
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def sphere_points(size, R):
    seed = secrets.randbits(32)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    # distance from centre
    r = rng.random(size)
    r = r * R
    # polar angle
    theta = rng.random(size)
    theta = np.arccos(1. - 2. * theta)
    # Azimuthal angle
    phi = rng.random(size)
    phi = 2. * np.pi * phi
    # Transform polar -> cartesian
    x, y, z = pol2cart(r, theta, phi)
    return x, y, z
# note for theta angle: using the inverse of the cumulative distribution function (CDF),
# the variable 1 - 2 * np.random.rand() generates random numbers in the range [-1, 1].
# By subtracting 1 and taking the arccosine (np.arccos), we map these values to the interval [0, π],
# which corresponds to the polar angle in spherical coordinates. Alternatively you can generate a random number
#between 0 and  1 and then multiply by pi

N = 1e4
R = 5
x, y, z = sphere_points(int(N), R)

for i in range(len(x)):
    print(np.sqrt(x[i]**2 + y[i]**2 + z[i]**2))

# Creating figure
fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection="3d")

# Creating plot
# Add x, y gridlines
ax.grid(b=True, color='grey',
        linestyle='-.', linewidth=0.3,
        alpha=0.2)

# Creating color map
my_cmap = plt.get_cmap('hsv')

# Creating plot
sctt = ax.scatter3D(x, y, z, alpha=0.8, c=(x + y + z), cmap=my_cmap,marker='^')
# more simply if you don't want a color map: ax.scatter3D(x, y, z, color = "green") e togli anche my_cmpa e fig.colorbar
plt.title("Random sphere points")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)
# show plot
plt.show()


"""Rejection method"""

def generate_points_rejection_method(num_points, radius):
    x, y, z = np.zeros(num_points), np.zeros(num_points), np.zeros(num_points)

    for i in range(num_points):
        # Generate random Cartesian coordinates
        x[i] = np.random.uniform(-radius, radius)
        y[i] = np.random.uniform(-radius, radius)
        z[i] = np.random.uniform(-radius, radius)

        # Reject points outside the sphere
        while x[i] ** 2 + y[i] ** 2 + z[i] ** 2 > radius ** 2:
            x[i] = 2 * radius * (np.random.rand() - 0.5)
            y[i] = 2 * radius * (np.random.rand() - 0.5)
            z[i] = 2 * radius * (np.random.rand() - 0.5)

    return x, y, z


# Parameters
num_points = 10 ** 4
radius = 5

# Generate points using rejection method
x_rej, y_rej, z_rej = generate_points_rejection_method(num_points, radius)

# Creating figure
fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection="3d")

# Creating plot
# Add x, y gridlines
ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.3, alpha=0.2)

# Creating color map
my_cmap = plt.get_cmap('hsv')

# Creating plot
sctt = ax.scatter3D(x_rej, y_rej, z_rej, alpha=0.8, c=(x + y + z), cmap=my_cmap,marker='^')
# more simply if you don't want a color map: ax.scatter3D(x, y, z, color = "green") e togli anche my_cmpa e fig.colorbar
plt.title("Random sphere points")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)
# show plot
plt.show()
