import numpy as np

def abs_cos18_potential(x, l, alpha=1):
    """
    Calculate the absorbing potential.

    Parameters:
    x (float array-like): The input value(s) for which the potential is calculated.
    l (float): Coordinate limit of the grid.
    alpha (float, optional): Absorbing factor for the potential. Default is 1.

    Returns:
    complex array-like: The calculated potential as a complex number.
    """
    return -1j * alpha * (1 - np.power(np.abs(np.cos(np.pi * x / l / 2)), 1 / 8))

class Hydrogen:
    def __init__(self, charge=1):
        """
        Initialize a Hydrogen atom.

        Parameters:
        charge (int): The charge of the nucleus divided by the elementary charge.
        """
        self.q = charge

    def energy(self, n=1):
        """
        Calculate the energy of the Hydrogen atom for a given principal quantum number n.
        The energy is given in atomic units (a.u.).

        Parameters:
        n (int): The principal quantum number.

        Returns:
        float: The energy of the Hydrogen atom for the given principal quantum number.
        """
        return 0.5 * self.q**2 / n**2

    def potential(self, x, eps=1e-2):
        """
        Calculate the electrostatic potential.
        The potential is given in atomic units (a.u.).

        Parameters:
        x (np.array): The distance |x| from the nucleus.
        eps (float, optional): A small value to avoid division by zero. Default is 1e-2.

        Returns:
        np.array: The value of the potential at the given distance.
        """
        return -self.q / np.sqrt(x**2 + eps**2)

    def ground_state_energy(self):
        """
        Calculate the ground state energy of the Hydrogen atom.
        The energy is given in atomic units (a.u.).

        Returns:
        float: The ground state energy of the Hydrogen atom.
        """
        return self.energy(1)

    def ground_state_wavefunction(self, x):
        """
        Calculate the ground state wavefunction of the Hydrogen atom at a distance |x| from the nucleus.
        The wavefunction is given in arbitrary units.

        Parameters:
        x (np.array): The distance |x| from the nucleus.

        Returns:
        np.array: The value of the wavefunction at the given distance.
        """
        a = 1 / self.q  # Bohr radius in atomic units (a.u.)
        return 1 / np.sqrt(a) * np.exp(-np.abs(x) / a)