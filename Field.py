import numpy as np

class Field:
    """
    Field class
    """

    class Pulse:
        """
        Pulse class representing a single pulse with Gaussian envelope.
        """
        def __init__(self, parameters):
            """
            Initialize a Pulse instance.

            Parameters: np.array of 
            duration (float): The duration of the pulse in ???
            amplitude (float): The amplitude of the pulse in ???
            frequency (float): The frequency of the pulse in a. u.
            phase (float): The phase of the pulse.
            """
            self.tau = parameters[0]  # Duration of the pulse
            self.a = parameters[1]  # Amplitude of the pulse
            self.w = parameters[2]  # Frequency of the pulse
            self.phi = parameters[3]  # Phase of the pulse
        
        def __call__(self, time, Type='All'):
            """
            Calculate the pulse value at a given time.

            Parameters:
            time (np.array): The time at which to evaluate the pulse in a. u.
            Type (str): The type of the pulse component to return ('Real', 'Imag', or 'Abs').

            Returns:
            np.array: The value of the pulse at the given time.
            """
            if Type == 'All':
                return self.a * np.exp(-4 * np.log(2) * time**2 / self.tau**2) * (np.cos(self.w * time + self.phi) + 1j * np.sin(self.w * time + self.phi))
            if Type == 'Real':
                return self.a * np.exp(-4 * np.log(2) * time**2 / self.tau**2) * np.cos(self.w * time + self.phi)
            if Type == 'Imag':
                return self.a * np.exp(-4 * np.log(2) * time**2 / self.tau**2) * np.sin(self.w * time + self.phi)
            if Type == 'Abs':
                return self.a * np.exp(-4 * np.log(2) * time**2 / self.tau**2)
        
        def get_pulse_parameters(self):
            """
            Get the parameters of the pulse.

            Returns:
            dict: A dictionary containing the parameters of the pulse.
            """
            return {'duration': self.tau, 'amplitude': self.a, 'frequency': self.w, 'phase': self.phi}


    class MultiPulse:
        """
        MultiPulse class representing a combination of multiple pulses.
        """
        def __init__(self, parameters):
            """
            Initialize a MultiPulse instance.

            Parameters: np.array of list of
            durations (np.array): The durations of the pulses in ???
            amplitudes (np.array): The amplitudes of the pulses in ???
            frequencies (np.array): The frequencies of the pulses in a.u
            phases (np.array): The phases of the pulses.
            """

            self.pulses = [Field.Pulse(params) for params in parameters]
        
        def __call__(self, time, Type='Real'):
            """
            Calculate the combined pulse value at a given time.

            Parameters:
            time (np.array): The time at which to evaluate the combined pulse in ???
            Type (str): The type of the pulse component to return ('Real', 'Imag', or 'Abs').

            Returns:
            np.array: The value of the combined pulse at the given time.
            """
            if Type == 'Real':
                return sum(pulse(time, Type) for pulse in self.pulses)
            if Type == 'Imag':
                return sum(pulse(time, Type) for pulse in self.pulses)
            if Type == 'Abs':
                return sum(pulse(time, Type) for pulse in self.pulses)
            
        def get_pulses_parameters(self):
            """
            Get the parameters of all pulses.

            Returns:
            list of dict: A list of dictionaries containing the parameters of each pulse.
            """
            return [{'duration': pulse.tau, 'amplitude': pulse.a, 'frequency': pulse.w, 'phase': pulse.phi} for pulse in self.pulses]
