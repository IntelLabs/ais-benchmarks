import numpy as np
import time

from numpy import array as t_tensor
from sampling_methods.base import CSamplingMethod
from distributions.CMultivariateNormal import CMultivariateNormal
from utils.plot_utils import plot_pdf


class CMixturePMC(CSamplingMethod):
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
        """
        super(self.__class__, self).__init__(space_min, space_max)
        self.K = params["K"]            # in the paper N
        self.N = params["N"]            # in the paper D
        self.J = params["J"]            # in the paper t
        self.sigma = params["sigma"]
        self.proposals = []             # q_d(\theta_d)
        self.wproposals = []            # \alpha_d
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

    def sample(self, n_samples):
        res = t_tensor([])
        for _ in range(n_samples):
            idx = np.argmax(np.random.multinomial(1, self.wproposals))  # Select the sampling proposal
            q = self.proposals[idx]
            x = q.sample(1)                                   # Sample from the sampling proposal
            res = np.concatenate((res, x)) if res.size else x
        return res

    def logprob(self, s):
        prob = np.zeros_like(s.flatten())
        for alpha, q in zip(self.wproposals, self.proposals):
            pval = alpha * q.logprob(s)
            prob = prob + pval.flatten()
            # prob = prob + pval if np.all(pval > 0) else prob
            self._num_q_evals += 1
        return prob

    def prob(self, s):
        prob = np.zeros_like(s.flatten())
        for alpha, q in zip(self.wproposals, self.proposals):
            pval = alpha * q.prob(s)
            prob = prob + pval.flatten()
            self._num_q_evals += 1
        return prob

    def adapt(self, samples, weights, posteriors):

        # Update all N proposals and proposal weights with the M-PMC update rule
        for d in range(self.N):
            # Update mixture weights (alpha_d)
            self.wproposals[d] = np.sum(weights * posteriors[d])

            # Update proposal parameters (gaussian proposal)
            mu = np.sum(weights * posteriors[d] * samples) / self.wproposals[d]
            cov = (samples - mu).T @ (samples - mu)
            sigma = np.sum(weights * posteriors[d] * cov, axis=1) / self.wproposals[d]
            # if np.any(np.isnan(sigma)):
            sigma = np.ones_like(sigma) * 0.01

            self.proposals[d].set_moments(mu, np.diag(sigma))

        # Renormalize proposals weights
        self.wproposals = self.wproposals / self.wproposals.sum()

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

                # Compute mixture posteriors (eq. 7)
                for d in range(self.N):
                    alpha_d = self.wproposals[d]
                    q_d = self.proposals[d].prob(new_samples[i])
                    rho[d][i] = (alpha_d * q_d) / q_x

            new_weights = new_weights / new_weights.sum()

            # Adaptation
            self.adapt(new_samples, new_weights, rho)

            self.samples = np.concatenate((self.samples, new_samples)) if self.samples.size else new_samples
            self.weights = np.concatenate((self.weights, new_weights)) if self.weights.size else new_weights

            elapsed_time = time.time() - t_ini

        return self.samples, self.weights

    def draw(self, ax):
        res = []
        for q in self.proposals:
            res.extend(ax.plot(q.mean.flatten(), 0, "gx", markersize=20))
            res.extend(plot_pdf(ax, q, self.space_min, self.space_max,
                                alpha=1.0, options="r--", resolution=0.01, scale=1/len(self.proposals)))

        # res.extend(ax.plot(q.mean.flatten(), 0, "gx", markersize=20, label="$\mu_n$"))
        res.extend(plot_pdf(ax, q, self.space_min, self.space_max, label="$q_n(x)$",
                            alpha=1.0, options="r--", resolution=0.01, scale=1/len(self.proposals)))

        for s, w in zip(self.samples, self.weights):
            res.append(ax.vlines(s, 0, w, "g", alpha=0.1))

        res.append(ax.vlines(s, 0, w, "g", alpha=0.1, label="$w_k = \pi(x_k) / \\frac{1}{N}\sum_{n=0}^N q_n(x_k)$"))

        res.extend(plot_pdf(ax, self, self.space_min, self.space_max, alpha=1.0, options="r-", resolution=0.01, label="$q(x)$"))

        return res
