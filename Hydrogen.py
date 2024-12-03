import numpy as np

class Hydrogen:
    def __init__(self, charge=1):
        """
        Initialize a Hydrogen atom

        Parameters:
        charge (int): The charge of the nucleus devided by the elementary charge.
        """
        self.q = charge

    def energy(self, n=1):
        """
        Calculate the energy of the Hydrogen atom for a given principal quantum number n.
        The energy is given in electron volts (eV).

        Parameters:
        n (int): The principal quantum number.

        Returns:
        float: The energy of the Hydrogen atom for the given principal quantum number.
        """
        return 0.5 * self.q**2 / n**2 # a. u.
    
    def potential(self, x ,eps= 1e-2):
        """
        Calculate the electrostatic potential.
        The potential is given in electron volts (eV).

        Parameters:
        x (np.array): The distance |x| from the nucleus.

        Returns:
        np.array: The value of the potential at the given distance.
        """
        return -self.q / np.sqrt(x**2 + eps**2) # a. u.
    
    def ground_state_energy(self):
        """
        Calculate the ground state energy of the Hydrogen atom.
        The energy is given in electron volts (eV).
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
        a = 1 / self.q # Bohr radius i a. u. (1 a. u. = 1 Bohr radius for q=1)
        return 1/np.sqrt(a) * np.exp(-np.abs(x) / a) # a. u.s