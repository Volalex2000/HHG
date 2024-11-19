# in test mode

import matplotlib.pyplot as plt
import numpy as np
import pywt

def __out_signal_calculation(x, wavefunction, potential, field):
    dV_dx = np.gradient(potential)
    a = np.trapz(-wavefunction * dV_dx * np.conj(wavefunction), x, axis=0) + field
    a = np.real(a + field)
    return a  # a. u.

def __wavelet_transform(a, FW_frequency, t0=0, tau=620.35, max_harm_order=100):
    scales = np.linspace(0, max_harm_order, 2 * max_harm_order)
    wavelet = pywt.ContinuousWavelet(f"cmor{tau:.3f}-{t0:.3f}")
    sampling_period = 2 * np.pi / FW_frequency
    A, frequencies = pywt.cwt(a, scales, wavelet, sampling_period)
    return A, frequencies

def __calculate_cutoff(E, W, Z):
    Up = np.sum(E**2 / W**2) / 4
    const = 3.17  # ??? (constant for cutoff calculation)
    Ip = Z**2 / 2  # for hydrogen-like atom
    return Ip + const * Up

def plot_HH_spectrum(x, wavefunction, parameters, potential, field, Z):
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
    A, frequencies = __wavelet_transform(a, FW_frequency)
    plt.figure()
    plt.plot(frequencies, np.log2(A))
    cutoff = __calculate_cutoff(parameters[1], parameters[2], Z)
    plt.axvline(x=cutoff / FW_frequency, color='r', linestyle='--', label=f'Cutoff: {cutoff:.2f}')
    plt.legend()
    plt.xlabel('Frequency, harmonic order')
    plt.ylabel('Log2(Amplitude)')
    plt.title('HH Spectrum')
    plt.show()

def imshow_time_frequency_characteristics(x, wavefunction, parameters, potential, field):
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
    A, frequencies = __wavelet_transform(a, np.min(parameters[2]))
    plt.figure()
    plt.imshow(np.log2(A), aspect='auto', extent=[0, len(wavefunction), np.min(frequencies), np.max(frequencies)])
    plt.xlabel('Time, a. u.')
    plt.ylabel('Frequency, harmonic order')
    plt.title('Time-Frequency characteristics')
    plt.show()
