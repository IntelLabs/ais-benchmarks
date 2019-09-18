import numpy as np
import time

from numpy import array as t_tensor
from sampling_methods.base import CSamplingMethod
from distributions.CMultivariateNormal import CMultivariateNormal
from utils.plot_utils import plot_pdf


class CDeterministicMixtureAIS(CSamplingMethod):
    def __init__(self, space_min, space_max, params):
        """
        Implementation of Deterministic Mixture Adaptive Importance Sampling algorithm
        :param space_min: Space lower boundary
        :param space_max: Space upper boundary
        :param params:
            - K: Number of samples per proposal
            - N: Number of proposals
            - J: Maximum number of iterations when sampling
        """
        super(self.__class__, self).__init__(space_min, space_max)
        self.K = params["K"]
        self.N = params["N"]
        self.J = params["J"]
        self.sigma = params["sigma"]
        self.proposals = []
        self.samples = t_tensor([])
        self.weights = t_tensor([])
        self.reset()

    def reset(self):
        # Initialize samples
        self.samples = t_tensor([])
        self.weights = t_tensor([])

        # Reset the proposals
        self.proposals = []
        for _ in range(self.N):
            prop_center = np.random.uniform(self.space_min, self.space_max)
            prop_d = CMultivariateNormal(prop_center, np.diag(t_tensor([self.sigma] * len(self.space_max))))
            self.proposals.append(prop_d)

    def sample(self, n_samples):
        raise NotImplementedError

    def prob(self, s):
        prob = 0
        for q in self.proposals:
            prob += q.prob(s)

        return prob / len(self.proposals)

    def resample(self, samples, weights):

        norm_weights = weights / weights.sum()

        # Update all N proposals
        for prop_d in self.proposals:
            idx = np.random.multinomial(1, norm_weights)
            new_mean = samples[idx == 1]
            prop_d.mean = new_mean

    def get_acceptance_rate(self):
        return t_tensor([1])

    def importance_sample(self, target_d, n_samples, timeout=60):
        elapsed_time = 0
        t_ini = time.time()

        while n_samples > len(self.samples) and elapsed_time < timeout:
            # Generate K samples from each proposal
            new_samples = t_tensor([])
            for q in self.proposals:
                for _ in range(self.K):
                    s = q.sample()
                    new_samples = np.concatenate((new_samples, s)) if new_samples.size else s

            # Weight samples
            new_weights = t_tensor([])
            for s in new_samples:
                w = target_d.prob(s) / self.prob(s)
                new_weights = np.concatenate((new_weights, w)) if new_weights.size else w

            # Adaptation
            self.resample(new_samples, new_weights)

            self.samples = np.concatenate((self.samples, new_samples)) if self.samples.size else new_samples
            self.weights = np.concatenate((self.weights, new_weights)) if self.weights.size else new_weights

            elapsed_time = time.time() - t_ini

        return self.samples, self.weights / self.weights.sum()

    def draw(self, ax):
        res = []
        for q in self.proposals:
            res.extend(ax.plot(q.mean, 0, "gx", markersize=20))
            res.extend(plot_pdf(ax, q, self.space_min, self.space_max,
                                alpha=1.0, options="r--", resolution=0.01, scale=1/len(self.proposals)))

        for s, w in zip(self.samples, self.weights):
            res.append(ax.vlines(s, 0, w, "g", alpha=0.1))

        res.extend(plot_pdf(ax, self, self.space_min, self.space_max, alpha=1.0, options="r-", resolution=0.01))

        return res
