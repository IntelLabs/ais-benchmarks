import numpy as np
import time

from numpy import array as t_tensor
from sampling_methods.base import CMixtureISSamplingMethod
from distributions.parametric.CMultivariateNormal import CMultivariateNormal
from distributions.mixture.CMixtureModel import CMixtureModel


class CLayeredAIS(CMixtureISSamplingMethod):
    def __init__(self, params):
        """
        Implementation of Deterministic Mixture Adaptive Importance Sampling algorithm
        :param params:
            - space_min: Space lower boundary
            - space_max: Space upper boundary
            - dims: Number of dimensions
            - K: Number of samples per proposal
            - N: Number of proposals
            - J: Maximum number of iterations when sampling
            - L: Number of MCMC moves during the proposal adaptation
            - sigma: Initial sigma for the mixture components
            - mh_sigma: Sigma for the transition distribution used for the mcmc adaptation steps
        """
        super(self.__class__, self).__init__(params)
        self.K = params["K"]
        self.N = params["N"]
        self.J = params["J"]
        self.L = params["L"]
        self.sigma = params["sigma"]
        self.mhsigma = params["mh_sigma"]
        self.mhproposal = CMultivariateNormal({"mean": np.zeros_like(self.space_max),
                                               "sigma": np.diag(t_tensor([self.mhsigma] * len(self.space_max)))})
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

        # Generate the mixture model induced by the LAIS proposals
        self.model = CMixtureModel(self.proposals, t_tensor([1 / len(self.proposals)] * len(self.proposals)))

    def mcmc_mh(self, x, prop_d, target_d, n_steps):
        for _ in range(n_steps):
            old_val = target_d.prob(x)
            self._num_pi_evals += 1
            x_delta = self.mhproposal.sample()[0]  # Discard batch dimension from batch sampler
            self._num_q_samples += 1
            x_hat = np.clip(x + x_delta, self.space_min, self.space_max)
            new_val = target_d.prob(x_hat)
            self._num_pi_evals += 1
            alpha = np.random.rand()
            ratio = new_val / old_val

            x = x_hat if ratio > alpha else x
            prop_d.set_moments(x, np.diag(t_tensor([self.sigma] * len(self.space_max))))
        return prop_d.mean

    def resample(self, target_d):
        # Update all N proposals by performing L MCMC steps
        for prop_d in self.proposals:
            new_mean = self.mcmc_mh(prop_d.mean, prop_d, target_d, self.L)
            prop_d.set_moments(new_mean, np.diag(t_tensor([self.sigma] * len(self.space_max))))

    def importance_sample(self, target_d, n_samples, timeout=60):
        elapsed_time = 0
        t_ini = time.time()

        while n_samples > len(self.samples) and elapsed_time < timeout:
            # Generate K samples from each proposal
            new_samples = t_tensor([])
            for q in self.proposals:
                for _ in range(self.K):
                    s = q.sample()[0]           # Samplers generate samples in a batch. Here we are generating just
                    self._num_q_samples += 1    # one sample and discard the batch dimension
                    new_samples = np.vstack((new_samples, s)) if new_samples.size else s

            # Weight samples
            new_weights = t_tensor([])
            for s in new_samples:
                w = target_d.prob(s) / self.prob(s)
                self._num_pi_evals += 1
                new_weights = np.concatenate((new_weights, w.reshape(-1))) if new_weights.size else w.reshape(-1)

            # Adaptation
            self.resample(target_d)

            self.samples = np.concatenate((self.samples, new_samples)) if self.samples.size else new_samples
            self.weights = np.concatenate((self.weights, new_weights)) if self.weights.size else new_weights

            elapsed_time = time.time() - t_ini

        return self.samples, self.weights / self.weights.sum()

    def _update_model(self):
        pass
