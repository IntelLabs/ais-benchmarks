import numpy as np
from sampling_experiments.distributions.CMixtureModel import CMixtureModel
from sampling_experiments.distributions.CMultivariateNormal import CMultivariateNormal


class CGaussianMixtureModel:
    def __init__(self, mu, sigma, weights=None):
        self.models = []
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
