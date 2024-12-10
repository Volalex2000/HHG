# in test mode

import matplotlib.pyplot as plt
import numpy as np

def __out_signal_calculation(x, wavefunction, potential, field):
    dV_dx = np.gradient(potential, x)
    a = np.trapz(-wavefunction * dV_dx[np.newaxis,:] * np.conj(wavefunction), x[np.newaxis,:], axis=1)
    a = np.real(a + field)
    return a  # a. u.

def __wavelet_transform(a, t, FW_frequency, t0=0, tau=620.35, max_harm_order=100, fast=False):
    scales = FW_frequency * np.arange(1, max_harm_order)

    if fast:
        X = scales[np.newaxis,np.newaxis,:] * (t[np.newaxis,:,np.newaxis] - t[:,np.newaxis,np.newaxis])
        wavelet = np.sqrt(scales[np.newaxis,np.newaxis,:] / tau) * np.exp(-X**2 / (2 * tau**2) + 1j * X)
        Int = a[:,np.newaxis,np.newaxis] * wavelet
        A[i] = np.trapz(Int, t, axis=0)
    else:
        A = np.zeros((len(t), len(scales)), dtype=complex)
        for i in range(len(t)):
            X = scales[np.newaxis, :] * (t[i] - t[:,np.newaxis])
            wavelet = np.sqrt(scales[np.newaxis,:] / tau) * np.exp(-X**2 / (2 * tau**2) + 1j * X)
            Int = a[:,np.newaxis] * wavelet
            A[i] = np.trapz(Int, t, axis=0)

    return A, scales

def __calculate_cutoff(E, W, Z):
    Up = np.sum(E**2 / W**2) / 4
    const = 3.17  # ??? (constant for cutoff calculation)
    Ip = Z**2 / 2  # for hydrogen-like atom
    return Ip + const * Up

def plot_HH_spectrum(x, t, wavefunction, parameters, potential, field, Z):
    """
    Plots the High Harmonic (HH) spectrum of a given signal.

    Args:
        x (array-like): The x-axis values for the signal.
        wavefunction (array-like): The wavefunction data to be analyzed.
        parameters (list): Parameters of the experiment.
        potential (array-like): The potential values used in the signal calculation.
        field (float): The field value used in the signal calculation.

    Returns:
        None: This function does not return any value. It generates a plot of the HH spectrum.
    """
    a = __out_signal_calculation(x, wavefunction, potential, field)
    FW_frequency = np.min(parameters[2])
    A, frequencies = __wavelet_transform(a, t, FW_frequency)
    print(A)
    plt.figure()
    plt.plot(frequencies / FW_frequency, np.log2(np.abs(A[0,:])))
    cutoff = __calculate_cutoff(parameters[1], parameters[2], Z)
    plt.axvline(x = cutoff / FW_frequency, color='r', linestyle='--', label=f'Cutoff: {cutoff:.2f}')
    plt.legend()
    plt.xlabel('Frequency, harmonic order')
    plt.ylabel('Log2(Amplitude)')
    plt.title('HH Spectrum')
    plt.show()

def imshow_time_frequency_characteristics(x, t, wavefunction, parameters, potential, field):
    """
    Display the time-frequency characteristics of a given signal using wavelet transform.

    Args:
        x (array-like): The x-axis values for the signal.
        wavefunction (array-like): The wavefunction data to be analyzed.
        parameters (list or array-like): Parameters of the experiment.
        potential (array-like): The potential values used in the signal calculation.
        field (float): The field value used in the signal calculation.

    Returns:
        None: This function does not return any value. It displays a plot of the time-frequency characteristics.
    """
    a = __out_signal_calculation(x, wavefunction, potential, field)
    FW_frequency = np.min(parameters[2])
    A, frequencies = __wavelet_transform(a, t, FW_frequency)
    plt.figure()
    plt.imshow(np.log2(np.abs(A)), aspect='auto', extent=[0, len(wavefunction), np.min(frequencies), np.max(frequencies)])
    plt.xlabel('Time, a. u.')
    plt.ylabel('Frequency, harmonic order')
    plt.title('Time-Frequency characteristics')
    plt.show()
