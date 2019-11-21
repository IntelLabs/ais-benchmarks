import numpy as np
from distributions.CMixtureModel import CMixtureModel
from distributions.CMultivariateNormal import CMultivariateNormal


class CGaussianMixtureModel:
    def __init__(self, mu, sigma, weights=None):
        self.models = []
        self.mu = mu
        self.sigma = sigma
        self.dims = len(mu[0])
        for mean, cov in zip(mu, sigma):
            self.models.append(CMultivariateNormal(mean, np.diag(cov)))

        if weights is None:
            self.weights = (1.0 / len(mu)) * np.array([1]*len(mu))
        else:
            self.weights = weights

        self.gmm = CMixtureModel(self.models, self.weights)

    def logprob(self, samples):
        return self.gmm.logprob(samples)

    def prob(self, samples):
        return self.gmm.prob(samples)

    def sample(self, nsamples):
        return self.gmm.sample(nsamples)
