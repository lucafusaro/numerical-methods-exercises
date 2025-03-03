def discrete_fourier_transform(signal):
    """
    Compute the Discrete Fourier Transform (DFT) of a given signal.

    Parameters:
        signal (numpy.ndarray): Input signal.

    Returns:
        numpy.ndarray: DFT of the input signal.
    """
    N = len(signal)
    k = np.arange(N)
    n = k.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    return np.dot(e, signal)

ck1 = discrete_fourier_transform(y)
abso_ck1 = abs(ck1)