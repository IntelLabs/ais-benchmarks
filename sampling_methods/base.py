import numpy as np
from numpy import array as t_tensor
from abc import ABCMeta, abstractmethod


class CSamplingMethod(metaclass=ABCMeta):
    def __init__(self, space_min, space_max):
        assert space_max.shape == space_min.shape
        self.space_min = space_min
        self.space_max = space_max
        self._num_pi_evals = 0
        self._num_q_evals = 0
        self._num_q_samples = 0
        self.samples = t_tensor([])
        self.weights = t_tensor([])

    def reset(self):
        self._num_pi_evals = 0
        self._num_q_evals = 0
        self._num_q_samples = 0
        self.samples = t_tensor([])
        self.weights = t_tensor([])

    def get_acceptance_rate(self):
        assert self.samples.size, "Invalid number of samples to compute acceptance rate"
        return len(self.samples) / self._num_q_samples

    @property
    def num_proposal_samples(self):
        return self._num_q_samples

    @property
    def num_target_evals(self):
        return self._num_pi_evals

    @property
    def num_proposal_evals(self):
        return self._num_q_evals

    @abstractmethod
    def sample(self, n_samples):
        raise NotImplementedError

    @abstractmethod
    def importance_sample(self, target_d, n_samples):
        raise NotImplementedError

    @abstractmethod
    def draw(self, ax):
        pass


def make_grid(space_min, space_max, resolution):
    dim_range = space_max - space_min
    num_samples = (dim_range / resolution).tolist()
    for i in range(len(num_samples)):
        num_samples[i] = int(num_samples[i])
        if num_samples[i] < 1:
            num_samples[i] = 1

    dimensions = []
    shape = t_tensor([0] * len(num_samples))
    for i in range(len(num_samples)):
        dimensions.append(np.linspace(space_min[i], space_max[i], num_samples[i]))
        shape[i] = len(dimensions[i])

    samples = np.array(np.meshgrid(*dimensions)).T.reshape(-1, len(space_min))
    return samples, dimensions, shape


def grid_sample_distribution(dist, space_min, space_max, resolution):
    grid, dims, shape= make_grid(space_min, space_max, resolution)
    log_prob = dist.logprob(t_tensor(grid))
    return grid, log_prob, dims, shape


def uniform_sample_distribution(dist, space_min, space_max, nsamples):
    samples = np.random.uniform(space_min, space_max, size=(nsamples,len(space_max)))
    log_prob = dist.logprob(samples.reshape(nsamples,len(space_max)))
    return samples.reshape(nsamples, len(space_max)), log_prob


