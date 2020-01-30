import numpy as np
import time
from sampling_methods.base import CMixtureSamplingMethod
from utils.plot_utils import plot_pdf
from utils.plot_utils import plot_pdf2d
import distributions


# TODO: Implement convergence test and futher generate samples from the approximated distribution
class CNestedSampling(CMixtureSamplingMethod):
    def __init__(self, params):
        super(self.__class__, self).__init__(params)
        self.range = self.space_max - self.space_min

        self.proposal_dist = eval(params["proposal"])
        self.N = params["N"]
        self.bw = np.array([params["kde_bw"]])

        self.live_points = None
        self.Z = None
        self.L = None

        self.reset()

    def reset(self):
        self.live_points = np.random.uniform(0, 1, size=(self.N, len(self.space_max))) * self.range + self.space_min
        self.L = np.array([])
        self.Z = 0
        super(CNestedSampling, self).reset()

    def resample(self, sample, value, pdf, timeout):
        new_sample = np.clip(self.proposal_dist.sample() + sample, self.space_min, self.space_max)
        self._num_q_samples += 1
        new_value = pdf.prob(new_sample)
        self._num_pi_evals += 1
        elapsed_time = 0
        t_ini = time.time()
        # print("Ellipsoid volume: %f. Converged: " % ellipsoid.volume, not ellipsoid.volume > ellipsoid_converged_radius**self.ndims)
        # while value > new_value and elapsed_time < timeout and ellipsoid.volume > ellipsoid_converged_radius**self.ndims:
        while value > new_value and elapsed_time < timeout:
            new_sample = self.proposal_dist.sample() + sample
            self._num_q_samples += 1
            new_value = pdf.prob(new_sample)
            self._num_pi_evals += 1
            elapsed_time = time.time() - t_ini
        return new_sample, new_value

    def get_NESS(self):
        ESS = np.exp(-np.sum(self.weights * np.log(self.weights)))
        return ESS / len(self.samples)

    def importance_sample(self, target_d, n_samples, timeout=60):
        values = target_d.prob(self.live_points)
        self._num_pi_evals += len(self.live_points)

        for i in range(n_samples - len(self.samples)):
            # Add the point with lowest likelihood to the resulting sample set
            Lmin = np.min(values)
            min_idx = np.argmin(values)
            self.samples = np.vstack((self.samples, self.live_points[min_idx])) if self.samples.size else self.live_points[min_idx].reshape(1, -1)

            # Keep the likelihood value associated with the sample
            self.L = np.concatenate((self.L, np.array([Lmin]))) if self.L.size else np.array([Lmin])

            # Replace the point with lowest likelihood with a new sample from the proposal distribution
            # centered on a sample randomly select4ed from the live points
            self.live_points[min_idx], values[min_idx] = self.resample(self.live_points[np.random.randint(0, len(self.live_points))], Lmin, target_d, timeout)

        # Update Z (evidence)
        X = np.array([np.exp(-i / self.N) for i in range(len(self.samples))])
        W = np.array([X[i - 1] - X[i] for i in range(1, len(self.samples))])
        W = np.concatenate((np.array([X[0]]), W))
        Z = np.sum(self.L * W)

        self.weights = (self.L * W) / Z
        self._update_model()
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
        for sample in self.live_points:
            res.extend(ax.plot([sample[0]], [sample[1]], c="g", marker="o", alpha=0.2))
        res.extend(ax.plot([sample[0]], [sample[1]], c="g", marker="o", alpha=0.2, label="live points"))

        res.extend(plot_pdf2d(ax, self, self.space_min, self.space_max, alpha=0.5, resolution=0.02, label="$q(x)$"))
        return res

    # def draw2d(self, ax):
    #     res = []
    #     for sample in self.live_points:
    #         res.extend(ax.plot([sample[0]], [sample[1]], 0, c="g", marker="o", alpha=0.2))
    #     res.extend(ax.plot([sample[0]], [sample[1]], 0, c="g", marker="o", alpha=0.2, label="live points"))
    #
    #     res.extend(plot_pdf2d(ax, self, self.space_min, self.space_max, alpha=0.5, resolution=0.02, colormap=cm.viridis, label="$q(x)$"))
    #     return res