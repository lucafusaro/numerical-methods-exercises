import numpy as np

"""Considering density distribution of a singular isothermal sphere"""
def integrand(r):
    G = 6.67430e-11  #N*m^2/Kg^2
    sigma = 1e4  #m/s
    if r == 0:
        f = 4 * np.pi * sigma**2 / (2 * np.pi * G)
    else:
        f = 4 * np.pi * (r**2) * sigma**2 / (2 * np.pi * G * (r**2))
    # in realtà qui semplicemnte potremmo togliere r^2 in generale dato che proprio si semplifica
    return f

def trapz(integrand, a, b, N):
    sum_value = 0
    h = (b - a)/N
    for k in range(1, N):
        sum_value += integrand(a + k * h)
    I = h * (0.5 * integrand(a) + 0.5 * integrand(b) + sum_value)
    return I

M_sun = 1.98892e30  # kg
r_max = 3.086e17  #10 parsec in metri
N = 100
M = trapz(integrand, 0, r_max, N)
print("Mass obtained with isothermal sphere DDF M =", M, "kg")
print("In solar masses: M =", M/M_sun, "M_sun" )

"""NFW density profile, good description of DM halos in LamdaCDM model"""
def integrand2(r):
    ro_0 = 1e8   # Msun / Kpc^-3
    if r < 1e-30:
        f2 = 0
    else:
        f2 = 4 * np.pi * (r**2) * ro_0 / ((r / r_s) * (1 + (r/r_s))**2)
    return f2


r_s = 10  #kpc
r_max2 = 100 * r_s
N2 = 5000
M2 = trapz(integrand2, 0, r_max2, 5000)

print("Risultato dell'integrale con NFW profile:", M2, "M_sun")