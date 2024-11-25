import numpy as np
import matplotlib.pyplot as plt

def __ground_state_energy(q):
    return 0.5 * q**2

def sc_potential(x, q):
    """
    Calculate an approximate electrostatic potential of the Hydrogen atom at a distance |x| from the nucleus.
    The potential is given in a. u.

    Parameters:
    x (np.array): The distance |x| from the nucleus.

    Returns:
    np.array: The value of the potential at the given distance.
    """
    return -q / np.sqrt(2 + x**2) # a. u.

def sc_wavefunction(x, q):
        
    y = np.zeros(len(x))
        
    y[0] = 0
    y[1] = 0.01
    
    h = x[1]-x[0]
    b = h**2 / 12.
    V_array = sc_potential(x, q)
    gst = __ground_state_energy(q)

    for i in range(1, len(y)-1):
        y[i+1] = (2 * y[i] - y[i-1] + \
                  b * (V_array[i+1] + \
                       10 * (V_array[i] + gst * y[i]) + \
                       V_array[i-1] + gst * y[i-1])) / (1 - gst * b)
        
    # y = ( y + y[::-1] ) / 2.
    y /= np.sqrt(np.trapz(y**2, x))

    return y

def __plot_test_sc_wavefunction():
    """
    Plot the soft-core ground state wavefunction of the Hydrogen atom.

    Args:
        x (array-like): The x-axis values for the wavefunction.
        q (float): The charge of the nucleus.

    Returns:
        None: This function does not return any value. It generates a plot of the wavefunction.
    """
    x = np.linspace(-100, 100, 1000)
    y = sc_wavefunction(x, 1)
    plt.figure()
    plt.plot(x, y)
    plt.xlabel('Distance from the nucleus, a. u.')
    plt.ylabel('Wavefunction')
    plt.title('Soft-core ground state wavefunction')
    plt.show()

__plot_test_sc_wavefunction()