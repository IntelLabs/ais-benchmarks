import numpy as np
from distributions.distributions import CDistribution


class CMultivariateExponential(CDistribution):
    def __init__(self, params):
        self._check_param(params, "mean")

        params["type"] = "exp"
        params["family"] = "exponential"
        params["likelihood_f"] = self.prob
        params["loglikelihood_f"] = self.log_prob
        params["dims"] = len(params["mean"])

        # Clip the exp support at 4 means
        if "support" not in params:
            params["support"] = np.array([np.zeros_like(params["mean"]), 4 * params["mean"]])

        super(CMultivariateExponential, self).__init__(params)

        self.set_moments(np.array(params["mean"]))

    def set_moments(self, mean):
        self.mean = mean
        self.lmbda = 1 / mean
        self.llmbda = np.log(self.lmbda * np.e)

    def sample(self, n_samples=1):
        return np.random.exponential(self.mean, n_samples)

    def log_prob(self, samples):
        samples = self._check_shape(samples)

        res = np.zeros(len(samples))
        inliers = np.all(samples > 0, axis=1).flatten()
        res_in = -self.lmbda.reshape(self.dims, 1) * samples[inliers] * self.llmbda.reshape(self.dims, 1)
        res_in = np.sum(res_in, axis=1)
        res[inliers] = res_in.flatten()
        return res

    def prob(self, samples):
        res = np.zeros(len(samples))
        inliers = np.all(samples > 0, axis=1).flatten()
        res[inliers] = np.exp(self.log_prob(samples[inliers]).flatten())
        return res

    def condition(self, dist):
        raise NotImplementedError

    def marginal(self, dim):
        raise NotImplementedError

    def support(self):
        return self.support_vals


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    mean = np.array([1.])
    dist = CMultivariateExponential({"mean": mean})

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title('CMultivariateExponential({"mean": %s})' % str(mean))
    dist.draw(plt.gca())

    mean = np.array([1., 2.0])
    dist2 = CMultivariateExponential({"mean": mean})

    plt.subplot(2, 1, 2)
    plt.title('CMultivariateExponential({"mean": %s})' % str(mean))
    dist2.draw(plt.gca())
    plt.show()
