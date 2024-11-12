import numpy as np

class Hydrogen:
    def __init__(self, charge=1):
        """
        Initialize a Hydrogen atom with a given charge.
        By default, the charge is set to 1 (proton).
        """
        self.q = charge

    def energy(self, n=1):
        """
        Calculate the energy of the Hydrogen atom for a given principal quantum number n.
        The energy is given in electron volts (eV).
        """
        return -13.6 * self.q**2 / n**2 # eV
    
    def potential(self, r):
        """
        Calculate the electrostatic potential of the Hydrogen atom at a distance r from the nucleus.
        The potential is given in electron volts (eV).
        """
        return -self.q / (r + 1e-10) # eV
    
    def soft_core_potential(self, r):
        """
        Calculate an approximate electrostatic potential of the Hydrogen atom at a distance r from the nucleus.
        The potential is given in electron volts (eV).
        """
        return -self.q / np.sqrt(2 + r**2) # eV

    class GroundState:
        def __init__(self, hydrogen):
            """
            Initialize the GroundState with a reference to the Hydrogen atom.
            """
            self.hydrogen = hydrogen

        def energy(self):
            """
            Calculate the energy of the Hydrogen atom in the ground state (n=1).
            """
            return self.hydrogen.energy(n=1)

        def wavefunction(self, r):
            """
            Calculate the wavefunction of the Hydrogen atom in the ground state.
            The wavefunction is a function of the distance r from the nucleus.
            """
            a0 = 0.0529 # Bohr radius in nm
            return 1 / np.sqrt(np.pi * a0**3) * np.exp(-r / a0)

# Example usage:
# hydrogen = Hydrogen()
# ground_state = hydrogen.GroundState(hydrogen)
# print(ground_state.energy())
# print(ground_state.wavefunction(0.1))