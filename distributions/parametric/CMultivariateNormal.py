import numpy as np
from distributions.distributions import CDistribution


class CMultivariateNormal(CDistribution):
    """
    Multivariate Normal distribution. Also known as multivariate gaussian and abbreviated MVN, MVG. This is a
    multidimensional generalization of the Normal distribution.

    This is a parametric distribution, meaning that it can be explicitly defined by a set of parameters and has
    the following properties:
        - Closed form Probability Density Function
        - Closed form Cumulative Density Function
        - Closed form for Kullback-Leibler Divergence


    :param params: The distribution class is initialized by a dictionary of parameters which must contain (besides
    class specific parameters) at least the following parameters:
    - mean: Array of first moments for each dimension. i.e. array of means
    - support: PDF support. Namely, subspace containing all the probability density. The definite integral of the
               PDF over the support subspace is 1.
               WARNING!!: Infinite support distributions are not supported yet. TODO. Must add support for such feature.
    """

    def __init__(self, params):
        self._check_param(params, "mean")
        self._check_param(params, "sigma")

        params["type"] = "normal"
        params["family"] = "exponential"
        params["likelihood_f"] = self.prob
        params["loglikelihood_f"] = self.log_prob
        params["dims"] = len(params["mean"])

        # Clip the normal support at 4 sigmas
        if "support" not in params:
            # TODO: Compute the support accounting for the cross terms of the covariance
            params["support"] = np.array([params["mean"] - np.sqrt(np.diag(params["sigma"])) * 6.0,
                                          params["mean"] + np.sqrt(np.diag(params["sigma"])) * 6.0])

        super(CMultivariateNormal, self).__init__(params)

        self.set_moments(np.array(params["mean"]), np.array(params["sigma"]))

        assert self.cov.shape == (self.dims, self.dims)
        self.det = np.linalg.det(self.cov)
        assert self.det > 0

        self.term1 = - 0.5 * self.dims * np.log(np.pi * 2)

    def set_moments(self, mean, cov):
        assert cov.shape == (self.dims, self.dims)

        self.det = np.linalg.det(cov)
        assert self.det > 0

        self.mean = mean
        self.cov = cov
        self.loc = self.mean
        self.scale = self.cov
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
        return (self.term1 + self.term2 + term3).reshape(len(samples), 1)

    def prob(self, samples):
        return np.exp(self.log_prob(samples))

    def condition(self, dist):
        raise NotImplementedError

    def marginal(self, dim):
        raise NotImplementedError

    def support(self):
        return self.support_vals

    def draw(self, ax, resolution=.01, label=None, color=None):
        super(CMultivariateNormal, self).draw(ax, resolution, label, color)


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    mean = np.array([0.0])
    cov = np.diag([0.1])
    dist = CMultivariateNormal({"mean": mean, "sigma": cov})

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title('CMultivariateNormal({"mean": %s, "sigma": %s})' % (str(mean), str(cov)))
    dist.draw(plt.gca())

    mean = np.array([.0, .0])
    cov = np.array([[0.1, .2], [.0, .2]])
    dist2 = CMultivariateNormal({"mean": mean, "sigma": cov})

    plt.subplot(2, 1, 2)
    plt.title('CMultivariateNormal({"mean": %s, "sigma": %s})' % (str(mean), str(cov)))
    dist2.draw(plt.gca())
    plt.show(block=True)
