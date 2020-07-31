import numpy as np
from sampling_methods.base import make_grid
from distributions.mixture.CGaussianMixtureModel import CGaussianMixtureModel


def generateRandomGMM(space_min, space_max, num_gaussians, sigma_min=(0.04, 0.04, 0.04), sigma_max=(0.05, 0.05, 0.05)):
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
    [means, dim, shape] = make_grid(space_min, space_max, delta)
    covs = np.array(np.ones_like(means) * sigma)

    gmm = CGaussianMixtureModel(means, covs)
    return gmm
