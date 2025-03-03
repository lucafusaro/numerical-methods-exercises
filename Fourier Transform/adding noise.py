"""We now add noise to the image, besides the blurring.
If we deconvolve the PSF, in this case we incorretly deconvolve also the noise.
This can lead to catastrophic effects"""

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def display(original, filtered):
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    axs[0].set_title('Before')
    axs[0].imshow(original)
    axs[1].imshow(filtered)
    axs[1].set_title('After')
    # remove the x and y ticks
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


image = np.genfromtxt('blur.txt')
K, L = image.shape

print(K, L)

# Adding noise
sigma = 100
frac = 0.


def add_noise(image, std, highpass):
    # Adding noise to the image
    # Generating Gaussian noise in pixel space
    K, L = image.shape
    mean = 0
    seed = 2938645
    np.random.seed(seed)
    noise = np.random.normal(mean, std, size=(K, L))
    # Removing low frequency noise components in Fourier space
    nk = np.fft.rfft2(noise)
    keep_fraction = highpass
    nk[0:int(K * highpass)] = 0
    nk[int(K * (1 - highpass)):] = 0
    nk[:, 0:int(K * highpass)] = 0
    # Fourier transforming back to get filtered noise field
    # in pixel space
    noise = np.fft.irfft2(nk)
    plt.imshow(abs(nk))
    return image + noise


image2 = add_noise(image, sigma, frac)

compare = display(image, image2)

"Point spread function"
sigma = 23
size=1024

def PSF(sigma,size):
    # Point spread function sampled in image grid
    def gauss(x,y,sigma):
        norm = 1./(2.*sigma**2)
        f = norm*np.exp(-(x**2 + y**2)/(2.*sigma**2))
        return f
    gauss = np.vectorize(gauss)
    x = np.linspace(0,size,size)
    y = np.linspace(0,size,size)
    X,Y = np.meshgrid(x,y)
    W = gauss(X,Y,sigma) + gauss(X-size,Y,sigma) + gauss(X,Y-size,sigma) + gauss(X-size,Y-size,sigma)
    return W

W = PSF(sigma,size)

#plt.imshow(W,cmap='Greys')
#plt.title('PSF (with periodic boundaries)')
#plt.show()

"Deconvolution"
ck_W = np.fft.rfft2(W)
ck_img = np.fft.rfft2(image2)

K, L = ck_W.shape
# Deconvolving PSF by division in Fourier space
ck_unblur = np.copy(ck_img)
for i in range(K):
    for j in range(L):
        if (abs(ck_W[i, j]) > 1e-3):
            ck_unblur[i, j] = ck_img[i, j] / (ck_W[i, j])

# Fourier transforming back to get the unblurred image
img_unblur = np.fft.irfft2(ck_unblur).real

# Plotting
compare = display(image2, img_unblur)