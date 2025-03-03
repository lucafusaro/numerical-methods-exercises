"""The file sunspots.txt contains the observed number of sunspots on the Sun for each month
since January 1749. The file contains two columns: column 0 is the month and column 1
is the number of sunspots per each month.
1) Plot the number of sunspots as a function of the month. Get a by estimate of the lenght of the cycle.
2) Write a script to perform the discete Fourier transf, DFT (use equation) and FFT (use np func)
of the  number of sunspots as a function of the month. Compare perfomances, DFT should be much slower than the FFT
3)Plot |ck|^2 as a function of k.
4)Find power spectrum peak for k>0. What does |c(0)|^2 represents?
Calculate period T = N/k associated with the largest |ck|, compare with by eye estiamate
This gives you the sunspot periodicity (t about 10.9 years)
5) Try to denoise the data assuming that there is just white noise. Plot denoised signal."""

import numpy as np
import matplotlib.pyplot as plt

"""def ck(y_values):
    for i in range(y_values):
        ck += y_values[i] * np.exp(-j * (2*np.pi*/))"""

"1) Read and plot"
month = np.genfromtxt('sunspots.txt', usecols=(0,))
sunspot = np.genfromtxt('sunspots.txt', usecols=(1,))
# alternativly: spots = np.genfromtxt('sunspots.txt')
# so for the months for example N = len(spots[:,0]) (N number of data = number of months)
plt.plot(month, sunspot)
plt.xlabel("Month", size=14)
plt.ylabel("Sunspot number", size=14)
plt.show()

"Zooming to get a by eye estimate of the period"
plt.xlabel('Month', size = 14)
plt.ylabel('Number of sunspots',size = 14)
plt.xlim(2000,2500)
plt.plot(month, sunspot)
plt.show()
"We get an estimate of 100-140"

"Fourier transform"
"""By plotting |c(k)|^2 we would see a large peak for k=0. 
This is just a constant offset (monopole), giving the average value 
of sunspots over the entire period of observation.
 Since we are interested in cycles, we remove it"""
ck = np.fft.rfft(sunspot)
ck[0] = 0
abso_ck = abs(ck)
power_ck = abso_ck**2
"Plot |ck|^2 vs k"
plt.plot(power_ck)
plt.xlabel('k', size=14)
plt.ylabel('$|c(k)|^2$', size=14)
plt.xlim(-1, 200)
plt.show()

"""Finding maximum of power spectrum and plotting corresponding mode,
 in comparison with the observed data"""
kmax = np.argmax(abso_ck)
filter = np.zeros_like(ck)
filter[kmax] = ck[kmax]
trend = np.fft.irfft(filter)

plt.plot(trend)
plt.plot(sunspot)
plt.xlabel('Months')
plt.show()

"""Getting period corresponding to frequency of maximum"""
N = len(month)
T = (N / kmax) / 12   #period in years, remember frequency is kmax/N
print("The period, ie the sunspot periodicity is about", T, "(yrs)")
