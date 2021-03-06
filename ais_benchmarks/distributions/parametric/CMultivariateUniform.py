import numpy as np
from ais_benchmarks.distributions.distributions import CDistribution


class CMultivariateUniform(CDistribution):
    def __init__(self, params):
        self._check_param(params, "center")
        self._check_param(params, "radius")

        params["type"] = "uniform"
        params["family"] = "uniform"
        params["likelihood_f"] = self.prob
        params["loglikelihood_f"] = self.log_prob
        params["dims"] = len(params["center"])
        params["support"] = [params["center"] - params["radius"], params["center"] + params["radius"]]
        super(CMultivariateUniform, self).__init__(params)
        self.loc = params["center"]
        self.scale = params["radius"]
        self.dims = len(self._loc)
        # If there is only one value for the radius. All dimensions use the same radius.
        if len(self._scale) == 1:
            self.volume = (self._scale * 2) ** self.dims
        elif len(self._scale) == self.dims:
            self.volume = np.prod([self._scale * 2])
        else:
            raise ValueError("Radius dimensionality not valid. It has to be 1 dim (all dimensions will use the same \
                             radius) or N dim, where N is the dimensionality")

        self.prob_val = 1 / self.volume
        self.logprob_val = np.log(self.prob_val)

    def sample(self, n_samples=1):
        minval = self._loc - self._scale
        maxval = self._loc + self._scale
        res = np.random.uniform(low=minval, high=maxval, size=(n_samples, self.dims))
        return res

    def log_prob(self, samples):
        samples = self._check_shape(samples)

        min_val = self._loc - self._scale
        max_val = self._loc + self._scale
        # Select the inliers if all the coordinates are in range
        inliers = np.all(np.logical_and(min_val < samples, samples <= max_val), axis=1)
        res = np.full(len(samples), self.logprob_val)
        res[np.logical_not(inliers.flatten())] = -np.inf
        return res

    def prob(self, samples):
        samples = self._check_shape(samples)

        min_val = self._loc - self._scale
        max_val = self._loc + self._scale

        # Select the inliers if all the coordinates are in range
        inliers = np.all(np.logical_and(min_val < samples, samples <= max_val), axis=1)
        res = np.full(len(samples), self.prob_val)
        res[np.logical_not(inliers.flatten())] = 0
        return res

    def condition(self, dist):
        raise NotImplementedError

    def integral(self, a, b, nsamples=None):
        assert np.all(a < b)
        ini = np.maximum(a, self.support_vals[0])
        end = np.minimum(b, self.support_vals[1])
        return (1 / self.volume) * np.prod((end-ini))

    def marginal(self, dim):
        raise NotImplementedError


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    center = np.array([0.0])
    radius = np.array([0.5])
    dist = CMultivariateUniform({"center": center, "radius": radius})

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title('CMultivariateUniform({"center": %s, "radius": %s})' % (str(center), str(radius)))
    dist.draw(plt.gca())

    center = np.array([0.0, 0.1])
    radius = np.array([0.5, 0.2])
    dist2d = CMultivariateUniform({"center": center, "radius": radius})

    plt.subplot(2, 1, 2)
    plt.title('CMultivariateUniform({"center": %s, "radius": %s})' % (str(center), str(radius)))
    dist2d.draw(plt.gca())
    plt.show(block=True)
