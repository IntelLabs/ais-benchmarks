import numpy as np
from distributions.base import CDistribution
from distributions.CMixtureModel import CMixtureModel
from distributions.CMultivariateNormal import CMultivariateNormal


class CGaussianMixtureModel(CDistribution):
    def __init__(self, params):
        """
        :param params: A dictionary containing the distribution parameters. Must define the following parameters:
                params["dims"]
                params["support"]
                params["means"]
                params["sigmas"]
                params["weights"]
        """
        params["type"] = "GMM"
        params["family"] = "mixture"
        params["likelihood_f"] = self.prob
        params["loglikelihood_f"] = self.log_prob
        super(CGaussianMixtureModel, self).__init__(params)

        self.models = []
        self.means = params["means"]
        self.sigmas = params["sigmas"]
        for mean, cov in zip(self.means, self.sigmas):
            self.models.append(CMultivariateNormal(mean, np.diag(cov)))

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

    def integral(self, a, b):
        raise NotImplementedError


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    means = np.array([[0.0], [0.7], [-0.5]])
    covs = np.array([[0.1], [0.05], [0.5]])
    weights = np.array([.4, .2, .4])

    dist = CGaussianMixtureModel({"means": means, "sigmas": covs, "weights": weights, "dims": 1, "support": [-3, 3]})

    plt.figure()
    dist.draw(plt.gca())
    plt.show(True)

