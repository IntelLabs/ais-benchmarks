import numpy as np
from sampling_experiments.sampling_methods.base import CSamplingMethod


class CGridSampling(CSamplingMethod):
    def __init__(self, space_min, space_max):
        super(self.__class__, self).__init__(space_min, space_max)
        self.range = space_max - space_min
        self.ndims = len(self.range)
        self.integral = 0
        self.cell_volume = 0
        self.resolution = 1

    def sample(self, n_samples):
        samples_per_dim = n_samples ** (1/self.ndims)
        resolution = self.range / samples_per_dim
        samples, dims = self.make_grid(self.space_min, self.space_max, resolution)
        return samples

    def sample_with_likelihood(self, pdf, n_samples):
        samples = self.sample(n_samples)
        values = pdf.log_prob(samples)
        self.integral = np.sum(np.exp(values) * self.cell_volume)
        return samples, values

    def make_grid(self, space_min, space_max, resolution):
        self.resolution = resolution
        self.cell_volume = np.prod(resolution)
        dim_range = space_max - space_min
        num_samples = (dim_range / resolution).tolist()
        for i in range(len(num_samples)):
            if num_samples[i] < 1 or num_samples[i]!=num_samples[i]:
                num_samples[i] = 1
            else:
                num_samples[i] = int(num_samples[i])

        dimensions = []
        for i in range(len(num_samples)):
            dimensions.append(np.linspace(space_min[i], space_max[i], num_samples[i]))

        samples = np.array(np.meshgrid(*dimensions)).T.reshape(-1, len(space_min))
        return samples, dimensions
