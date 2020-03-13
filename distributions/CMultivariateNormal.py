import numpy as np
from distributions.base import CDistribution


class CMultivariateNormal(CDistribution):
    def __init__(self, params):
        params["type"] = "normal"
        params["family"] = "exponential"
        params["likelihood_f"] = self.prob
        params["loglikelihood_f"] = self.log_prob
        super(CMultivariateNormal, self).__init__(params)

        assert cov.shape == (self.dims, self.dims)

        self.det = np.linalg.det(cov)
        assert self.det > 0

        self.set_moments(params["mean"], params["sigma"])

        self.term1 = - 0.5 * self.dims * np.log(np.pi * 2)

    def set_moments(self, mean, cov):
        assert cov.shape == (self.dims, self.dims)

        self.det = np.linalg.det(cov)
        assert self.det > 0

        self.mean = mean
        self.cov = cov
        self.inv_cov = np.linalg.inv(cov)
        self.log_det = np.log(self.det)
        self.term2 = - 0.5 * self.log_det

    def sample(self, n_samples=1):
        return np.random.multivariate_normal(self.mean.flatten(), self.cov, size=n_samples)

    def log_prob(self, samples):
        if len(samples.shape) == 1:
            samples = samples.reshape(1, self.dims, 1)
        elif len(samples.shape) == 2:
            samples = samples.reshape(len(samples), self.dims, 1)
        else:
            raise ValueError("Shape of samples does not match self.dims")

        diff = self.mean.reshape(1, self.dims, 1) - samples
        term3 = -0.5 * (np.transpose(diff, axes=(0, 2, 1)) @ self.inv_cov @ diff)
        return (self.term1 + self.term2 + term3).reshape(len(samples))

    def prob(self, samples):
        return np.exp(self.log_prob(samples))

    def condition(self, dist):
        raise NotImplementedError

    def marginal(self, dim):
        raise NotImplementedError

    def integral(self, a, b):
        raise NotImplementedError

    def support(self):
        return self.support_vals


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    mean = np.array([0.0])
    cov = np.diag([1.0])
    support = np.array([-10, 10])

    dist = CMultivariateNormal({"mean": mean, "sigma": cov, "dims": 1, "support": support})

    plt.figure()
    dist.draw(plt.gca())
    plt.show(True)
