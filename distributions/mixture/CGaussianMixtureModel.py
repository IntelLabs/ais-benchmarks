import numpy as np
from distributions.distributions import CDistribution
from distributions.mixture.CMixtureModel import CMixtureModel
from distributions.parametric.CMultivariateNormal import CMultivariateNormal


class CGaussianMixtureModel(CDistribution):
    def __init__(self, params):
        """
        :param params: A dictionary containing the distribution parameters. Must define the following parameters:
                params["support"]
                params["means"]
                params["sigmas"]
                params["weights"]
        """
        self._check_param(params, "means")
        self._check_param(params, "sigmas")

        params["type"] = "GMM"
        params["family"] = "mixture"
        params["likelihood_f"] = self.prob
        params["loglikelihood_f"] = self.log_prob
        params["dims"] = len(params["means"][0])
        super(CGaussianMixtureModel, self).__init__(params)

        self.models = []
        self.means = params["means"]
        self.sigmas = params["sigmas"]
        for mean, cov in zip(self.means, self.sigmas):
            self.models.append(CMultivariateNormal({"mean": mean, "sigma": np.diag(cov)}))

        if "weights" in params.keys():
            self.weights = params["weights"]
        else:
            self.weights = (1.0 / len(self.means)) * np.array([1] * len(self.means))

        self.gmm = CMixtureModel(self.models, self.weights)

    def log_prob(self, samples):
        return self.gmm.log_prob(samples)

    def prob(self, samples):
        return self.gmm.prob(samples)

    def sample(self, nsamples=1):
        return self.gmm.sample(nsamples)

    def condition(self, dist):
        raise NotImplementedError

    def marginal(self, dim):
        raise NotImplementedError

    def integral(self, a, b, nsamples=None):
        raise NotImplementedError


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    means = np.array([[0.0], [0.7], [-0.5]])
    covs = np.array([[0.1], [0.05], [0.5]])
    weights = np.array([.4, .2, .4])
    support = np.array([-2, 2])

    # 1D GMM
    dist = CGaussianMixtureModel({"means": means, "sigmas": covs, "weights": weights, "dims": 1, "support": support})
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title('CGaussianMixtureModel({"means": %s, "sigmas": %s, "weights": %s})' %
              (str(dist.means), str(dist.sigmas), str(dist.weights)))
    dist.draw(plt.gca())
    plt.show(block=False)

    # 2D GMM
    means = np.array([[.0, .2], [.7, .2], [-.5, -.5]])
    covs = np.array([[.1, .3], [.05, .2], [.1, .05]])
    weights = np.array([.4, .2, .4])
    support = np.array([[-2, -2], [2, 2]])

    dist2 = CGaussianMixtureModel({"means": means, "sigmas": covs, "weights": weights, "support": support})
    plt.subplot(2, 1, 2)
    plt.title('CGaussianMixtureModel({"means": %s, "sigmas": %s, "weights": %s})' %
              (str(dist2.means), str(dist2.sigmas), str(dist2.weights)))
    dist2.draw(plt.gca())
    plt.show(block=True)
