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
        """
        return -13.6 * self.q**2 / n**2 # eV
    
    def potential(self, x):
        """
        Calculate the electrostatic potential.
        The potential is given in electron volts (eV).

        Parameters:
        x (float): The distance |x| from the nucleus.
        """
        return -self.q / (np.abs(x) + 1e-10) # eV
    
    def soft_core_potential(self, x):
        """
        Calculate an approximate electrostatic potential of the Hydrogen atom at a distance |x| from the nucleus.
        The potential is given in electron volts (eV).

        Parameters:
        x (float): The distance |x| from the nucleus.
        """
        return -self.q / np.sqrt(2 + x**2) # eV