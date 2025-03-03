"""Files piano.txt and trumpet.txt contain a sampling at 44100 Hz of a single note,
 played by a piano and a trumpet. Load the waveforms and plot them
Apply fft and plot the first 10000 coefficients of the power spectrum, for both instruments
Which note where the two instruments playing? [hint: the note middle C ('do') has a frequency of 216 Hz]"""

import numpy as np
import matplotlib.pyplot as plt

"""Downloading and plotting signal:"""
trumpet = np.genfromtxt('trumpet.txt')
plt.plot(trumpet)
plt.title('Trumpet waveform', size = 14)
plt.xlabel('Samples', size = 12)
plt.show()

piano = np.genfromtxt('piano.txt')
plt.plot(piano)
plt.title('Piano waveform', size = 14)
plt.xlabel('Samples', size = 12)
plt.show()

"FT and power spectrum:"
# Trumpet
ck_trumpet = np.fft.rfft(trumpet)
power_trumpet = abs(ck_trumpet)**2
plt.xlim(0, 10000)
plt.plot(power_trumpet)
plt.yscale('log')
plt.title('Trumpet', size = 14)
plt.xlabel('k', size = 14)
plt.show()

# Piano
ck_piano = np.fft.rfft(piano)
power_piano = abs(ck_piano)**2
plt.xlim(0, 10000)
plt.yscale('log')
plt.title('Piano', size = 14)
plt.xlabel('k', size = 14)
plt.plot(power_piano)
plt.show()

"""What note playing: to do that: find the maximum of the power spectrum and convert to frequency,
  knowing that the sampling frequency was 44100 Hz)"""
# Trumpet
sampling_freq = 44100
kmax = np.argmax(power_trumpet)
f = (kmax/(len(ck_trumpet)))*44100
print('Trumpet largest frequency component (in units of 261 Hz), f = ', f/261)

# Piano
kmax = np.argmax(power_piano)
f = (kmax/(len(ck_piano)))*44100
print('Piano largest frequency component (in units of 261 Hz), f = ', f/261)
