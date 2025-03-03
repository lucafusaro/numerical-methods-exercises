"""In this example we download a blurred image from the file blur.txt.
Knowing that the point spread function is a Gaussian with 𝜎 = 25,
we unblur it via PSF deconvolution, which is just a division by the PSF in Fourier space"""

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

"Reading file and displaying image"
image = np.genfromtxt('blur.txt')
plt.imshow(image)
plt.show()
K, L = image.shape

"Generating and plotting the PSF"
size = len(image)
sigma = 23

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

plt.imshow(W,cmap='Greys')
plt.title('PSF (with periodic boundaries)')
plt.show()


#######################################################
# Calculating PSF with loops, instead of vectorizing

#def PSF(x,y,sigma):
#    norm = 1./(2.*sigma**2)
#    f = norm*np.exp(-(x**2 + y**2)/(2.*sigma**2))
#    return f

#W = np.empty_like(image)
#size = len(W)
#sigma = 23

# Calculating PSF with loops
#for x in range (len(W)):
#    for y in range(len(W)):
#        W[x,y] = PSF(x,y,sigma) + PSF(x-size,y,sigma) + PSF(x,y-size,sigma) + PSF(x-size,y-size,sigma)

#plt.imshow(W,cmap='Greys')
#plt.title('PSF (with periodic boundaries)')
#plt.show()

"Deconvolving the PSF and comparing images before and after unblurring"
ck_img = np.fft.rfft2(image)
ck_W = np.fft.rfft2(W)

K, L = ck_W.shape

# Deconvolving PSF by division in Fourier space
ck_unblur = np.copy(ck_img)
for i in range(K):
    for j in range(L):
        if (abs(ck_W[i, j]) > 1e-3):
            ck_unblur[i, j] = ck_img[i, j] / (ck_W[i, j])

# Fourier transforming back to get the unblurred image
img_unblur = np.fft.irfft2(ck_unblur)

# Plotting

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

compare = display(image, img_unblur)
