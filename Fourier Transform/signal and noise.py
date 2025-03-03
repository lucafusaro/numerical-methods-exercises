"""In this example we download from the file pitch.txt a signal made of
 a basic wave, in which we recognize some high freqency noise component
  and we perform basic denoising by Fourier transforming, removing
  the high-k coefficients and transforming back."""

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

"""Downloading and plotting signal:"""
pitch = np.genfromtxt('pitch.txt')

plt.plot(pitch)
plt.xlabel('$t/\Delta t$', size=16)
plt.ylabel("f", size=16)
plt.title("Signal")
plt.show()

"FT and power spectrum:"
ck = np.fft.rfft(pitch)

abso_ck = abs(ck)
power = (abs(ck))**2

"Plot |ck| vs k"
plt.plot(abso_ck)
plt.xlabel('k', size=14)
plt.ylabel('$|c(k)|$', size=14)
plt.show()

"Plot power spectrum"
plt.plot(power)
plt.xlabel('k', size=14)
plt.ylabel('$P(k) = |c(k)|^2$', size=14)
plt.yscale('log')
plt.show()

"Main spike"
kmax = np.argmax(power)
print("k associated to max |ck| kmax =", kmax)
v_max = kmax/len(pitch)
print("max frequency associated:", v_max)

"Denoising"
# Removing high frequency coefficients
Delta = 100
dck = ck.copy()
dck[kmax + Delta:] = 0.0
denoised_pitch = np.fft.irfft(dck)
plt.xlabel('$t/ \Delta t$', size=14)
plt.ylabel("f", size=14)
plt.title('$\Delta = $' + str(Delta), size=16)
plt.plot(pitch, label="signal")
plt.plot(denoised_pitch, label="signal denoised")
plt.legend()
plt.show()