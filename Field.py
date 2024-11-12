import numpy as np

class Field:
    """
    Field class
    """

    class Pulse:
        """
        Pulse class representing a single pulse with Gaussian envelope.
        """
        def __init__(self, duration, amplitude, frequency, phase=0):
            """
            Initialize a Pulse instance.

            Parameters:
            duration (float): The duration of the pulse.
            amplitude (float): The amplitude of the pulse.
            frequency (float): The frequency of the pulse.
            phase (float): The phase of the pulse.
            """
            self.tau = duration  # Duration of the pulse
            self.a = amplitude  # Amplitude of the pulse
            self.w = frequency  # Frequency of the pulse
            self.phi = phase  # Phase of the pulse
        
        def __call__(self, time, Type='Real'):
            """
            Calculate the pulse value at a given time.

            Parameters:
            time (float): The time at which to evaluate the pulse.
            Type (str): The type of the pulse component to return ('Real', 'Imag', or 'Abs').

            Returns:
            float: The value of the pulse at the given time.
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
        def __init__(self, durations, amplitudes, frequencies, phases=None):
            """
            Initialize a MultiPulse instance.

            Parameters:
            durations (list of float): The durations of the pulses.
            amplitudes (list of float): The amplitudes of the pulses.
            frequencies (list of float): The frequencies of the pulses.
            phases (list of float): The phases of the pulses.
            """
            if phases is None:
                phases = [0] * len(durations)
            self.pulses = [Field.Pulse(duration, amplitude, frequency, phase) for duration, amplitude, frequency, phase in zip(durations, amplitudes, frequencies, phases)]
        
        def __call__(self, time, Type='Real'):
            """
            Calculate the combined pulse value at a given time.

            Parameters:
            time (float): The time at which to evaluate the combined pulse.
            Type (str): The type of the pulse component to return ('Real', 'Imag', or 'Abs').

            Returns:
            float: The value of the combined pulse at the given time.
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
