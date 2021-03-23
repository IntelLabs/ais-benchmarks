import numpy as np
import time

from numpy import array as t_tensor
from ais_benchmarks.sampling_methods.base import CMixtureISSamplingMethod
from ais_benchmarks.distributions import CMultivariateNormal
from ais_benchmarks.distributions import CMixtureModel


class CDeterministicMixtureAIS(CMixtureISSamplingMethod):
    def __init__(self, params):
        """
        Implementation of Deterministic Mixture Adaptive Importance Sampling algorithm DM-PMC
        :param params:
            - space_min: Space lower boundary
            - space_max: Space upper boundary
            - dims: Number of dimensions
            - K: Number of samples per proposal
            - N: Number of proposals
            - J: Maximum number of iterations when sampling
        """
        super(self.__class__, self).__init__(params)
        self.K = params["K"]
        self.N = params["N"]
        self.J = params["J"]
        self.sigma = params["sigma"]
        self.proposals = []
        self.model = None
        self.reset()

    def reset(self):
        super(self.__class__, self).reset()

        # Reset the proposals
        self.proposals = []
        for _ in range(self.N):
            prop_center = np.random.uniform(self.space_min, self.space_max)
            prop_d = CMultivariateNormal({"mean": prop_center,
                                          "sigma": np.diag(t_tensor([self.sigma] * len(self.space_max)))})
            self.proposals.append(prop_d)

        # Generate the mixture model induced by the DM-AIS proposals
        self.model = CMixtureModel({"models": self.proposals,
                                    "weights": t_tensor([1 / len(self.proposals)] * len(self.proposals)),
                                    "dims": self.ndims, "support": [self.space_min, self.space_max]})

    def resample(self, samples, weights):
        # Normalize weights
        norm_weights = weights / weights.sum()

        # Update all N proposals with the DM-PMC update rule
        for prop_d in self.proposals:
            idx = np.random.multinomial(1, norm_weights)
            idx = np.argmax(idx)
            new_mean = samples[idx]
            prop_d.set_moments(new_mean, np.diag(t_tensor([self.sigma] * len(self.space_max))))

    def importance_sample(self, target_d, n_samples, timeout=60):
        elapsed_time = 0
        t_ini = time.time()

        while n_samples > len(self.samples) and elapsed_time < timeout:
            # Generate K samples from each proposal
            new_samples = t_tensor([])
            for q in self.proposals:
                for _ in range(self.K):
                    s = q.sample()[0]
                    self._num_q_samples += 1
                    new_samples = np.vstack((new_samples, s)) if new_samples.size else s

            # Weight samples
            new_weights = t_tensor([])
            for s in new_samples:
                w = target_d.prob(s) / self.prob(s)
                self._num_pi_evals += 1
                new_weights = np.concatenate((new_weights, w.reshape(-1))) if new_weights.size else w.reshape(-1)

            # Adaptation
            self.resample(new_samples, new_weights)

            self.samples = np.concatenate((self.samples, new_samples)) if self.samples.size else new_samples
            self.weights = np.concatenate((self.weights, new_weights)) if self.weights.size else new_weights

            elapsed_time = time.time() - t_ini

        return self.samples, self.weights / self.weights.sum()

    def _update_model(self):
        pass
