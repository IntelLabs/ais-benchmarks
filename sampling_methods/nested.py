import numpy as np
import time
from sampling_methods.base import CSamplingMethod
from distributions.CMultivariateNormal import CMultivariateNormal
from utils.plot_utils import plot_pdf
from utils.plot_utils import plot_pdf2d


class CNestedSampling(CSamplingMethod):
    def __init__(self, space_min, space_max, params):
        super(self.__class__, self).__init__(space_min, space_max)
        self.range = space_max - space_min

        self.proposal_dist = params["proposal"]
        self.N = params["N"]
        self.bw = np.array([params["kde_bw"]])

        # Obtain initial samples from a uniform prior distribution
        self.live_points = np.random.uniform(0, 1, size=(self.N, len(self.space_max))) * self.range + self.space_min

    def sample(self, n_samples):
        raise NotImplementedError

    def reset(self):
        self.live_points = np.random.uniform(0, 1, size=(self.N, len(self.space_max))) * self.range + self.space_min

    def logprob(self, samples):
        return np.log(self.prob(samples))

    def prob(self, samples):
        prob = np.zeros(len(samples))
        for x in self.samples:
            cov = np.ones(len(self.space_max)) * self.bw
            q = CMultivariateNormal(x, np.diag(cov))
            pval = q.prob(samples)
            prob = prob + pval.flatten()
            self._num_q_evals += 1
        return prob / len(self.samples)

    def resample(self, sample, value, pdf, timeout):
        new_sample = self.proposal_dist.sample() + sample
        self._num_q_samples += 1
        new_value = pdf.logprob(new_sample)
        self._num_pi_evals += 1
        elapsed_time = 0
        t_ini = time.time()
        while value > new_value and elapsed_time < timeout:
            new_sample = self.proposal_dist.sample() + sample
            self._num_q_samples += 1
            new_value = pdf.logprob(new_sample)
            self._num_pi_evals += 1
            elapsed_time = time.time() - t_ini
        return new_sample, new_value

    def importance_sample(self, target_d, n_samples, timeout=60):
        points = self.live_points
        values = target_d.logprob(points)

        L = np.zeros(n_samples)
        X = np.zeros(n_samples)
        W = np.zeros(n_samples)
        Z = 0
        for i in range(n_samples - len(self.samples)):
            L[i] = np.min(values)
            X[i] = np.exp(-i / self.N)
            W[i] = X[i-1] - X[i]
            Z = Z + L[i] * W[i]

            # Add the point with lowest likelihood to the resulting sample set
            min_idx = np.argmin(values)
            self.samples = np.vstack((self.samples, points[min_idx])) if self.samples.size else points[min_idx]

            # Replace the point with lowest likelihood with a new sample from the proposal distribution
            points[min_idx], values[min_idx] = self.resample(points[min_idx], L[i], target_d, timeout)

        self.weights = target_d.logprob(self.samples)
        return self.samples, self.weights

    def draw(self, ax):
        if len(self.space_max) == 1:
            return self.draw1d(ax)

        elif len(self.space_max) == 2:
            return self.draw2d(ax)
        return []

    def draw1d(self, ax):
        res = []
        for point in self.live_points:
            res.extend(ax.plot(point, 0.1, "go"))
        res.extend(ax.plot(point, 0.1, "go", label="live points"))
        res.extend(plot_pdf(ax, self, self.space_min, self.space_max, alpha=1.0, options="r-", resolution=0.01,label="$q(x)$"))
        return res

    def draw2d(self, ax):
        res = []
        for sample, type in zip(self.trajectory_samples, self.trajectory_types):
            if type == self.BURN_IN:
                res.extend(ax.plot([sample[0]], [sample[1]], 0, c="r", marker="o", alpha=0.2))
            elif type == self.REJECT:
                res.extend(ax.plot([sample[0]], [sample[1]], 0, c="r", marker=".", alpha=0.2))
            elif type == self.DECORRELATION:
                res.extend(ax.plot([sample[0]], [sample[1]], 0, c="g", marker=".", alpha=0.2))
            elif type == self.SAMPLE:
                res.extend(ax.plot([sample[0]], [sample[1]], 0, c="g", marker="o"))

        res.append(plot_pdf2d(ax, self, self.space_min, self.space_max, alpha=0.5, resolution=0.02, colormap=cm.viridis, label="$q(x)$"))
        res.extend(ax.plot([0], [0], 0, "ro", c="r", marker="o", label="burn-in"))
        res.extend(ax.plot([0], [0], 0, "r.", c="r", marker="x", label="rejected"))
        res.extend(ax.plot([0], [0], 0, "g.", c="g", marker="o", label="intermediate"))
        res.extend(ax.plot([0], [0], 0, "go", c="g", marker="x", label="sample"))
        return res