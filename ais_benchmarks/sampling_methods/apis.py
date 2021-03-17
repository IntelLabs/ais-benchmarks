import numpy as np
import time

from numpy import array as t_tensor
from sampling_methods.base import CMixtureSamplingMethod
from distributions.parametric.CMultivariateNormal import CMultivariateNormal
from distributions.mixture.CMixtureModel import CMixtureModel


# TODO: Implement this method properly from the paper
class CAdaptivePopulationIS(CMixtureSamplingMethod):
    def __init__(self, space_min, space_max, params):
        """
        Implementation of Adaptive Population Importance Sampling (APIS) algorithm
        :param space_min: Space lower boundary
        :param space_max: Space upper boundary
        :param params:
            - K: Number of samples per iteration
            - N: Number of proposal mixture components
            - J: Maximum number of iterations of sampling algorithm
            - sigma: Initial sigma for the mixture components
        """
        super(self.__class__, self).__init__(space_min, space_max)
        self.K = params["K"]
        self.N = params["N"]
        self.J = params["J"]
        self.sigma = params["sigma"]
        self.proposals = []
        self.wproposals = []
        self.model = None
        self.reset()
        raise NotImplementedError("This class is a tentative implementation. Is not finalized or tested")

    def reset(self):
        super(self.__class__, self).reset()

        # Reset the proposals
        self.proposals = []
        for _ in range(self.N):
            prop_center = np.random.uniform(self.space_min, self.space_max)
            prop_d = CMultivariateNormal(prop_center, np.diag(t_tensor([self.sigma] * len(self.space_max))))
            self.proposals.append(prop_d)

        # Set uniform weights
        self.wproposals = t_tensor([1/self.N] * self.N)

        # Generate the mixture model induced by the AMIS proposals
        self.model = CMixtureModel(self.proposals, self.wproposals)

    def adapt(self):
        samples = self.samples
        weights = self.weights

        # Update all N proposals with the ML for the mixture proposal (similar to M-PMC adapt rule)
        for d in self.N:
            # Update proposal parameters (gaussian proposal)
            # Mean eq. 14.2
            mu = np.sum(weights.reshape(-1, 1) * posteriors[d].reshape(-1, 1) * samples, axis=0) / self.wproposals[d]

            # Covariance eq. 14.3
            cov_val = (samples - mu).T @ (samples - mu)
            cov = np.zeros_like(cov_val)
            for i in range(self.K):
                cov += weights[i] * posteriors[d][i] * cov_val
            cov /= self.wproposals[d]

            if np.any(np.isnan(cov)):
                cov = np.diag(np.ones(self.dims) * 0.01)

            self.proposals[d].set_moments(mu, cov)

    def importance_sample(self, target_d, n_samples, timeout=60):
        elapsed_time = 0
        t_ini = time.time()

        while n_samples > len(self.samples) and elapsed_time < timeout:
            # Generate K samples and add them to the sample set (Generic AMIS algorithm. Step 1)
            new_samples = self.sample(self.K)
            self.samples = np.concatenate((self.samples, new_samples)) if self.samples.size else new_samples

            # For each sample in the sample set: compute importance weights. (Generic AMIS algorithm. Steps 2 and 3)
            new_weights = np.zeros(self.K)
            self.weights = np.concatenate((self.weights, new_weights)) if self.weights.size else new_weights
            for i in range(len(self.samples)):
                pi_x = target_d.prob(self.samples[i])
                q_x = self.prob(self.samples[i])
                self.weights[i] = pi_x / q_x if q_x > 0 else 0
                self._num_pi_evals += 1
                self._num_q_evals += 1

            # Adaptation (Generic AMIS algorithm. Step 4)
            self.adapt()

            elapsed_time = time.time() - t_ini

        return self.samples, self.weights / self.weights.sum()
