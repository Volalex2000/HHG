# in test mode

import matplotlib.pyplot as plt
import numpy as np

def cutoff(E, W, Z=1):
    Up = np.sum(E**2 / W**2) / 4
    const = 3.17  # constant for cutoff calculation
    Ip = Z**2 / 2  # for hydrogen-like atom
    return Ip + const * Up

def plot_HH_spectrum(i, A, FW, scales, parameters):
    """
    Plots the High Harmonic (HH) spectrum of a given signal.

    Args:
        ???
    Returns:
        None: This function does not return any value. It generates a plot of the HH spectrum.
    """
    plt.figure()
    plt.plot(scales / FW, np.log2(np.abs(A[i,:])))
    cutoff = cutoff(parameters[1], parameters[2])
    plt.axvline(x = cutoff / FW, color='r', linestyle='--', label=f'Cutoff: {cutoff:.2f}')
    plt.legend()
    plt.xlabel('Frequency, harmonic order')
    plt.ylabel('Log2(Amplitude)')
    plt.title('HH Spectrum')
    plt.show()

def imshow_time_frequency_characteristics(A, frequencies):
    """
    Display the time-frequency characteristics of a given signal using wavelet transform.

    Args:
        ???

    Returns:
        None: This function does not return any value. It displays a plot of the time-frequency characteristics.
    """
    plt.figure()
    plt.imshow(np.log2(np.abs(A)), aspect='auto', extent=[0, len(A), np.min(frequencies), np.max(frequencies)])
    plt.xlabel('Time, a. u.')
    plt.ylabel('Frequency, harmonic order')
    plt.title('Time-Frequency characteristics')
    plt.show()
