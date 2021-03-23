from numpy import array as t_tensor
from scipy import spatial
import numpy as np
from ais_benchmarks.distributions import CDistribution


class CNearestNeighbor(CDistribution):
    def __init__(self, params):
        self._check_param(params, "samples")
        self._check_param(params, "weights")

        params["type"] = "NN"
        params["family"] = "nonparametric"
        params["likelihood_f"] = self.prob
        params["loglikelihood_f"] = self.log_prob
        params["dims"] = len(params["samples"][0])
        super(CNearestNeighbor, self).__init__(params)

        self.samples = params["samples"]
        self.weights = params["weights"]
        self.kdtree = spatial.KDTree(samples)

    def log_prob(self, samples):
        dist, idx = self.kdtree.query(samples)
        return self.weights[idx]

    def sample(self, nsamples=1):
        idxs = np.random.multinomial(nsamples, self.weights)
        return self.samples[idxs]

    def prob(self, data):
        return np.exp(self.log_prob(data))

    def condition(self, dist):
        raise NotImplementedError

    def marginal(self, dim):
        raise NotImplementedError

    def integral(self, a, b, nsamples=None):
        raise NotImplementedError


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    samples = np.array([-.1, .3, .2, .25, .31]).reshape(5, 1)
    weights = np.random.rand(len(samples))
    support = [-1, 1]
    params = dict()
    params["samples"] = samples
    params["weights"] = weights / np.sum(weights)
    params["support"] = support
    dist = CNearestNeighbor(params)
    plt.figure()
    plt.subplot(2, 1, 1)
    dist.draw(plt.gca(), label=dist.type)
    plt.scatter(samples, weights)
    plt.legend()

    samples = np.array([[-.1, .2], [.3, -.4], [.2, .1], [.25, -.3], [.31, .7]]).reshape(5, 2)
    weights = np.random.rand(len(samples))
    support = [[-1, -1], [1, 1]]
    params = dict()
    params["samples"] = samples
    params["weights"] = weights / np.sum(weights)
    params["support"] = support
    dist = CNearestNeighbor(params)
    plt.subplot(2, 1, 2)
    dist.draw(plt.gca(), label=dist.type)
    plt.scatter(samples[:, 0], samples[:, 1])
    plt.legend()

    plt.show(block=True)
