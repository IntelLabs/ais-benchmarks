import numpy as np
from distributions.distributions import CDistribution
from distributions.distributions import CKernel
from distributions.mixture.CMixtureModel import CMixtureModel


class CKernelDensity(CDistribution):
    def __init__(self, params):
        self._check_param(params, "samples")
        self._check_param(params, "weights")
        self._check_param(params, "kernel_f")
        self._check_param(params, "kernel_bw")

        params["type"] = "KDE"
        params["family"] = "mixture"
        params["likelihood_f"] = self.prob
        params["loglikelihood_f"] = self.log_prob
        params["dims"] = len(params["samples"][0])

        super(CKernelDensity, self).__init__(params)

        self.samples = params["samples"]
        self.weights = params["weights"]
        self.kernel_f = params["kernel_f"]
        self.kernel_bw = params["kernel_bw"]
        self.model = None

        assert callable(self.kernel_f), "kernel must be callable"

        self.fit()

    def fit(self):
        models = []
        for x in self.samples:
            models.append(CKernel(x, self.kernel_bw, self.kernel_f))
        self.model = CMixtureModel(models, self.weights)

    def log_prob(self, data):
        return self.model.log_prob(data)

    def sample(self, nsamples=1):
        return self.model.sample(nsamples)

    def prob(self, data):
        return self.model.prob(data)

    def condition(self, dist):
        raise NotImplementedError

    def marginal(self, dim):
        raise NotImplementedError

    def integral(self, a, b, nsamples=None):
        raise NotImplementedError

    def support(self):
        return self.support_vals


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    samples = np.array([-.1, .3, .2, .25, .31]).reshape(5, 1)
    weights = np.ones(len(samples)) / len(samples)
    support = [-1, 1]

    params = dict()
    params["samples"] = samples
    params["weights"] = weights
    params["kernel_f"] = CKernel.kernel_epanechnikov
    params["kernel_bw"] = np.array([5])
    params["support"] = support
    dist = CKernelDensity(params)

    plt.figure()
    plt.subplot(2, 2, 1)
    dist.draw(plt.gca(), label=params["kernel_f"].__name__, n_points=1000)
    plt.legend()

    plt.subplot(2, 2, 2)
    params["kernel_f"] = CKernel.kernel_normal
    dist = CKernelDensity(params)
    dist.draw(plt.gca(), label=params["kernel_f"].__name__, n_points=1000)
    plt.legend()

    plt.subplot(2, 2, 3)
    params["kernel_f"] = CKernel.kernel_triangular
    dist = CKernelDensity(params)
    dist.draw(plt.gca(), label=params["kernel_f"].__name__, n_points=1000)
    plt.legend()

    plt.subplot(2, 2, 4)
    params["kernel_f"] = CKernel.kernel_uniform
    dist = CKernelDensity(params)
    dist.draw(plt.gca(), label=params["kernel_f"].__name__, n_points=1000)
    plt.legend()

    plt.show(block=True)
