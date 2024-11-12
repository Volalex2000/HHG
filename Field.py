import numpy as np

class Field():
    """
    Field class
    """

    class Pulse:
        """
        Pulse class
        """
        def __init__(self, duration, amplitude, frequency):
            self.tau = duration
            self.a = amplitude
            self.w = frequency
        
        def __call__(self, time, Type = 'Real'):
            if Type == 'Real':
                return self.a * np.exp(-4*np.log(2) * time**2/self.tau**2) * np.cos(self.w*time)
            if Type == 'Imag':
                return self.a * np.exp(-4*np.log(2) * time**2/self.tau**2) * np.sin(self.w*time)
            if Type == 'Abs':
                return self.a * np.exp(-4*np.log(2) * time**2/self.tau**2)


    class MultiPulse():
        """
        MultiPulse class
        """
        def __init__(self, durations, amplitudes, frequencies):
            self.pulses = [Field.Pulse(duration, amplitude, frequency) for duration, amplitude, frequency in zip(durations, amplitudes, frequencies)]
        
        def __call__(self, time, Type = 'Real'):
            if Type == 'Real':
                return sum([pulse(time, Type) for pulse in self.pulses])
            if Type == 'Imag':
                return sum([pulse(time, Type) for pulse in self.pulses])
            if Type == 'Abs':
                return sum([pulse(time, Type) for pulse in self.pulses])