import numpy as np

class CRipple:
    def __init__(self,amplitude=10.0, freq=10):
        self.freq = freq
        self.amplitude = amplitude

    def log_prob(self, x):
        if len(x.shape) == 1:
            x = x.reshape(1,-1)

        r = np.sum(x[:] * x[:], axis=1)

        assert not np.any(np.isnan(r))
        z = np.sin(r * self.freq) * self.amplitude + self.amplitude
        return np.log(z / (r+1))