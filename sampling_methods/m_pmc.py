import numpy as np
import time

from numpy import array as t_tensor
from sampling_methods.base import CMixtureSamplingMethod
from distributions.CMultivariateNormal import CMultivariateNormal
from distributions.CMixtureModel import CMixtureModel


class CMixturePMC(CMixtureSamplingMethod):
    def __init__(self, space_min, space_max, params):
        """
        Implementation of Mixture Particle Monte Carlo Adaptive Importance Sampling algorithm
        https://arxiv.org/pdf/0710.4242.pdf

        :param space_min: Space lower boundary
        :param space_max: Space upper boundary
        :param params:
            - K: Number of samples per proposal
            - N: Number of proposals
            - J: Maximum number of iterations when sampling
            - sigma: Initial sigma for the mixture components
        """
        super(self.__class__, self).__init__(space_min, space_max)
        self.K = params["K"]            # in the paper N
        self.N = params["N"]            # in the paper D
        self.J = params["J"]            # in the paper t
        self.sigma = params["sigma"]
        self.proposals = []             # q_d(\theta_d)
        self.wproposals = []            # \alpha_d
        self.model = None
        self.dims = len(space_max)
        self.reset()

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

        # Generate the mixture model induced by the M-PMC proposals
        self.model = CMixtureModel(self.proposals, self.wproposals)

    def adapt(self, samples, weights, posteriors):
        # Update all N proposals and proposal weights with the M-PMC update rule
        for d in range(self.N):
            # Update mixture weights (alpha_d) eq. 14.1
            self.wproposals[d] = np.sum(weights * posteriors[d])

            # Update proposal parameters (gaussian proposal)
            # Mean eq. 14.2
            mu = np.sum(weights.reshape(-1, 1) * posteriors[d].reshape(-1, 1) * samples, axis=0) / self.wproposals[d]

            # Covariance eq. 14.3
            cov_val = (samples - mu).T @ (samples - mu)
            cov = np.zeros_like(cov_val)
            for i in range(self.K):
                cov += weights[i] * posteriors[d][i] * cov_val
            cov /= self.wproposals[d]

            # Failsafe for collapsing or exploding covariances
            if np.any(np.isnan(cov)) or np.any(cov > 1):
                cov = np.diag(np.ones(self.dims) * 0.01)

            self.proposals[d].set_moments(mu, cov)

        # Renormalize proposals weights
        self.wproposals = self.wproposals / self.wproposals.sum()
        self.model.weights = self.wproposals

    def importance_sample(self, target_d, n_samples, timeout=60):
        elapsed_time = 0
        t_ini = time.time()

        while n_samples > len(self.samples) and elapsed_time < timeout:
            # Generate K samples
            new_samples = self.sample(self.K)

            # Compute normalized importance weights and posteriors
            rho = np.zeros((self.N, self.K))
            new_weights = np.zeros(self.K)

            # For each sample
            for i in range(self.K):
                pi_x = target_d.prob(new_samples[i])
                q_x = self.prob(new_samples[i])
                new_weights[i] = pi_x / q_x if q_x > 0 else 0

                self._num_pi_evals += 1
                self._num_q_evals += 1

                # Compute mixture posteriors (eq. 7)
                for d in range(self.N):
                    alpha_d = self.wproposals[d]
                    q_d = self.proposals[d].prob(new_samples[i])
                    rho[d][i] = (alpha_d * q_d) / q_x
                    self._num_q_evals += 1

            new_weights = new_weights / new_weights.sum()

            # Adaptation
            self.adapt(new_samples, new_weights, rho)

            self.samples = np.concatenate((self.samples, new_samples)) if self.samples.size else new_samples
            self.weights = np.concatenate((self.weights, new_weights)) if self.weights.size else new_weights

            elapsed_time = time.time() - t_ini

        return self.samples, self.weights

    def _update_model(self):
        pass
