import numpy as np
from distributions.base import CDistribution


class CMultivariateUniform(CDistribution):
    def __init__(self, params):
        params["type"] = "uniform"
        params["family"] = "uniform"
        params["likelihood_f"] = self.prob
        params["loglikelihood_f"] = self.log_prob
        super(CMultivariateUniform, self).__init__(params)
        self.center = params["center"]
        self.radius = params["radius"]
        self.dims = len(self.center)
        self.volume = np.prod(np.array([self.radius*2]*self.dims))

    def sample(self, n_samples=1):
        minval = self.center - self.radius
        maxval = self.center + self.radius
        # TODO: Fix the size problem here and sample in one instruction
        res = np.random.uniform(low=minval, high=maxval, size=None)
        for i in range(1, n_samples):
            res = np.vstack((res, np.random.uniform(low=minval, high=maxval, size=None)))
        return res.reshape(n_samples, self.dims)

    def log_prob(self, samples):
        return np.log(self.prob(samples))

    def prob(self, samples):
        if len(samples.shape) == 1:
            samples = samples.reshape(1, self.dims)
        elif len(samples.shape) == 2:
            samples = samples.reshape(len(samples), self.dims)
        else:
            raise ValueError("Shape of samples does not match self.dims")

        min_val = self.center - self.radius
        max_val = self.center + self.radius
        inliers = np.all(np.logical_and(min_val < samples, samples < max_val), axis=1)  # Select the inliers if all the coordinates are in range
        res = np.ones(len(samples)) / self.volume
        res[np.logical_not(inliers.flatten())] = 0
        return res

    def condition(self, dist):
        raise NotImplementedError

    def integral(self, a, b):
        raise NotImplementedError

    def marginal(self, dim):
        raise NotImplementedError


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    center = np.array([0.0])
    radius = np.array([1.0])
    support = np.array([-radius, radius])

    dist = CMultivariateUniform({"center": center, "radius": radius, "dims": 1, "support": support})

    plt.figure()
    dist.draw(plt.gca())
    plt.show(True)
