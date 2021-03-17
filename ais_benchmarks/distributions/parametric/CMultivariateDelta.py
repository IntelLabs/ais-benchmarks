import numpy as np
from distributions.distributions import CDistribution


class CMultivariateDelta(CDistribution):
    def __init__(self, params):
        self._check_param(params, "center")
        self._check_param(params, "support")

        params["type"] = "delta"
        params["family"] = "delta"
        params["likelihood_f"] = self.prob
        params["loglikelihood_f"] = self.log_prob
        params["dims"] = len(params["center"])
        super(CMultivariateDelta, self).__init__(params)
        self.center = params["center"]
        self.loc = self.center
        self.scale = None
        self.dims = len(self.center)

        self.prob_val = 1
        self.logprob_val = np.log(self.prob_val)

    def sample(self, n_samples=1):
        return np.full((n_samples, self.dims), self.center)

    def log_prob(self, samples):
        samples = self._check_shape(samples)

        # Select the inliers
        inliers = np.all(np.isclose(self.center, samples), axis=1)
        res = np.full(len(samples), -np.inf)
        res[inliers.flatten()] = np.inf
        return res

    def prob(self, samples):
        samples = self._check_shape(samples)

        # Select the inliers
        inliers = np.all(np.isclose(self.center, samples), axis=1)
        res = np.full(len(samples), 0.0)
        res[inliers.flatten()] = np.inf
        return res

    def condition(self, dist):
        raise NotImplementedError

    def integral(self, a, b, nsamples=None):
        assert np.all(a < b)
        if np.all(a < center) and np.all(center < b):
            return np.array([1])
        return np.array([0])

    def marginal(self, dim):
        raise NotImplementedError

    def draw(self, ax, resolution=.01, label=None, color=None):
        if self.dims == 1:
            ax.plot((self.center, self.center), (0, 1), label=label, color=color)
        elif self.dims == 2:
            ax.scatter(self.center[0], self.center[1], label=label, color=color)


if __name__ == "__main__":
    import matplotlib
    from matplotlib import pyplot as plt

    center = np.array([0.0])
    support = np.array([-0.5, 0.5])
    dist = CMultivariateDelta({"center": center, "support": support})
    test_points = np.array([center, center + 0.001, center - 0.001])
    print("Probs should be [inf, 0, 0] -> ", dist.prob(test_points))
    print("LogProbs should be [inf, -inf, -inf] -> ", dist.log_prob(test_points))
    print("Integral should be 1 -> ", dist.integral(center - 1, center + 1))
    print("Integral should be 0 -> ", dist.integral(center - 2, center - 1))
    print("Samples should be all %s -> " % str(center), dist.sample(10))

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title('CMultivariateDelta({"center": %s})' % str(center))
    dist.draw(plt.gca())

    center = np.array([0.0, 0.1])
    support = np.array([[-0.5, -0.2], [1.5, 1.2]])
    dist2d = CMultivariateDelta({"center": center, "support": support})
    test_points = np.array([center, center + 0.001, center - 0.001])
    print("Probs should be [inf, 0, 0] -> ", dist2d.prob(test_points))
    print("LogProbs should be [inf, -inf, -inf] -> ", dist2d.log_prob(test_points))
    print("Integral should be 1 -> ", dist2d.integral(center - 1, center + 1))
    print("Integral should be 0 -> ", dist2d.integral(center - 2, center - 1))
    print("Samples should be all %s -> " % str(center), dist2d.sample(10))

    plt.subplot(2, 1, 2)
    plt.title('CMultivariateDelta({"center": %s})' % str(center))
    dist2d.draw(plt.gca())
    plt.show(block=True)
