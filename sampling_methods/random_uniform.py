import numpy as np
from sampling_experiments.sampling_methods.base import CSamplingMethod


class CRandomUniformSampling(CSamplingMethod):
    def __init__(self, space_min, space_max):
        super(self.__class__, self).__init__(space_min, space_max)
        self.range = space_max - space_min

    def sample(self, n_samples):
        samples = np.random.uniform(0, 1, size=(n_samples,len(self.space_max)))
        return (samples * self.range) + self.space_min

    def sample_with_likelihood(self, pdf, n_samples):
        samples = self.sample(n_samples)
        values = pdf.log_prob(samples)
        return samples, values
