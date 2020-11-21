import numpy as np
from distributions.distributions import CDistribution


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
        self.center = params["center"]
        self.radius = params["radius"]
        self.loc = self.center
        self.scale = self.radius
        self.dims = len(self.center)
        # If there is only one value for the radius. All dimensions use the same radius.
        if len(self.radius) == 1:
            self.volume = (self.radius * 2) ** self.dims
        elif len(self.radius) == self.dims:
            self.volume = np.prod([self.radius * 2])
        else:
            raise ValueError("Radius dimensionality not valid. It has to be 1 dim (all dimensions will use the same \
                             radius) or N dim, where N is the dimensionality")

        self.prob_val = 1 / self.volume
        self.logprob_val = np.log(self.prob_val)

    def sample(self, n_samples=1):
        minval = self.center - self.radius
        maxval = self.center + self.radius
        res = np.random.uniform(low=minval, high=maxval, size=(n_samples, self.dims))
        return res

    def log_prob(self, samples):
        if len(samples.shape) == 1:
            samples = samples.reshape(1, self.dims)
        elif len(samples.shape) == 2:
            samples = samples.reshape(len(samples), self.dims)
        else:
            raise ValueError("Shape of samples does not match self.dims = %d" % self.dims)

        min_val = self.center - self.radius
        max_val = self.center + self.radius
        inliers = np.all(np.logical_and(min_val < samples, samples <= max_val), axis=1)  # Select the inliers if all the coordinates are in range
        res = np.full(len(samples), self.logprob_val)
        res[np.logical_not(inliers.flatten())] = -np.inf
        # return res.reshape(len(samples), 1)
        return res

    def prob(self, samples):
        if len(samples.shape) == 1:
            samples = samples.reshape(1, self.dims)
        elif len(samples.shape) == 2:
            samples = samples.reshape(len(samples), self.dims)
        else:
            raise ValueError("Shape of samples does not match self.dims = %d" % self.dims)

        min_val = self.center - self.radius
        max_val = self.center + self.radius
        inliers = np.all(np.logical_and(min_val < samples, samples <= max_val), axis=1)  # Select the inliers if all the coordinates are in range
        res = np.full(len(samples), self.prob_val)
        res[np.logical_not(inliers.flatten())] = 0
        # return res.reshape(len(samples), 1)
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
