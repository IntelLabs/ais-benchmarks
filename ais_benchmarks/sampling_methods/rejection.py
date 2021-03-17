import numpy as np
import time
from ais_benchmarks.distributions import *

from ais_benchmarks.sampling_methods.base import CMixtureSamplingMethod
from ais_benchmarks.utils.plot_utils import plot_pdf
from ais_benchmarks.utils.plot_utils import plot_pdf2d


class CRejectionSampling(CMixtureSamplingMethod):
    def __init__(self, params):
        super(self.__class__, self).__init__(params)
        self.proposal_dist = eval(params["proposal"])
        self.scaling = params["scaling"]
        self.bw = params["kde_bw"]

    def importance_sample(self, target_d, n_samples, timeout=60):
        t_ini = time.time()
        t_elapsed = time.time() - t_ini
        while t_elapsed < timeout and n_samples > len(self.samples):
            # Generate proposal samples from proposal distribution
            proposals = self.proposal_dist.sample(n_samples - len(self.samples))
            self._num_q_samples += n_samples

            # Obtain the sample likelihood for both the proposal and target distributions
            proposals_prob_q = self.proposal_dist.prob(proposals) * self.scaling
            self._num_pi_evals += n_samples
            proposals_prob_p = target_d.prob(proposals)
            self._num_q_evals += n_samples

            # Compute the acceptance ratio
            acceptance_ratio = proposals_prob_p / proposals_prob_q

            # Generate acceptance probability values from U[0,1)
            acceptance_prob = np.random.uniform(0, 1, size=n_samples - len(self.samples))

            # Obtain the indices of the accepted samples
            accept_idx = (acceptance_ratio > acceptance_prob).reshape(-1)

            self.samples = np.vstack((self.samples, proposals[accept_idx])) \
                if self.samples.size else proposals[accept_idx]

            self.weights = np.concatenate((self.weights, acceptance_ratio[accept_idx].reshape(-1))) \
                if self.weights.size else acceptance_ratio[accept_idx].reshape(-1)

            t_elapsed = time.time() - t_ini

        # Return the accepted samples
        self.weights = np.ones_like(self.weights) * 1 / len(self.weights)
        self._update_model()
        return self.samples, self.weights

    def draw(self, ax):
        if len(self.space_max) == 1:
            return self.draw1d(ax)

        elif len(self.space_max) == 2:
            return self.draw2d(ax)
        return []

    def prob(self, s):
        return self.proposal_dist.prob(s) * self.scaling

    def draw1d(self, ax):
        res = []
        res.extend(plot_pdf(ax, self, self.space_min, self.space_max, alpha=1.0, options="--", color="r",
                            resolution=0.01, label="$q(x)$ * scale"))
        # res.extend(plot_pdf(ax, self.proposal_dist, self.space_min, self.space_max, alpha=1.0, options="-", color="r",
        #                     resolution=0.01, label="$q(x)$"))
        res.extend(plot_pdf(ax, super(CRejectionSampling, self), self.space_min, self.space_max, alpha=1.0, options="-", color="g",
                            resolution=0.01, label="$KDE$"))
        return res

    def draw2d(self, ax):
        res = []
        res.extend(plot_pdf2d(ax, self, self.space_min, self.space_max, alpha=0.5, resolution=0.02, label="$q(x)$"))
        return res