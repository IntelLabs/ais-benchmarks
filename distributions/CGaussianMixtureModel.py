import numpy as np
from sampling_methods.base import make_grid
from distributions.CMixtureModel import CMixtureModel
from distributions.CMultivariateNormal import CMultivariateNormal


class CGaussianMixtureModel:
    def __init__(self, mu, sigma, weights=None):
        self.models = []
        self.mu = mu
        self.sigma = sigma
        for mean, cov in zip(mu, sigma):
            self.models.append(CMultivariateNormal(mean, np.diag(cov)))

        if weights is None:
            self.weights = (1.0 / len(mu)) * np.array([1]*len(mu))
        else:
            self.weights = weights

        self.gmm = CMixtureModel(self.models, self.weights)

    def log_prob(self, samples):
        return self.gmm.log_prob(samples)


def generateRandomGMM(space_min, space_max, num_gaussians, sigma_min=(0.04,0.04,0.04), sigma_max=(0.05,0.05,0.05)):
    means = []
    covs = []

    for _ in range(num_gaussians):
        mu = np.random.uniform(space_min, space_max)
        sigma = np.random.uniform(sigma_min, sigma_max)
        means.append(mu)
        covs.append(sigma)
    gmm = CGaussianMixtureModel(means, covs)
    return gmm


def generateEggBoxGMM(space_min, space_max, delta, sigma):
    ndims = len(space_min)
    [means, dim, shape] = make_grid(space_min, space_max, delta)
    covs = np.array(np.ones_like(means) * sigma)

    gmm = CGaussianMixtureModel(means, covs)
    return gmm
