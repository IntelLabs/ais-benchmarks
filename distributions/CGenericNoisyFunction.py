import numpy as np
from distributions.base import CDistribution


class GenericNoisyFunction(CDistribution):
    def __init__(self, params):
        super(GenericNoisyFunction, self).__init__(params)
        self.noise_level = params["noise"]
        self.model = params["model"]
        self.supp = params["support"]
        self.x = None

    def sample(self):
        noise = np.random.normal(0, self.noise_level, size=self.x.size).reshape(self.x.shape)
        fx = self.model(self.x) + noise
        return fx

    def logprob(self, x):
        if self.x is None:
            raise ValueError("GenericFunction prob or logprob cannot be evaluated w/o conditioning it first with the true x value.")

        if len(self.x.shape) == 1:
            dims = len(self.x)
            mu = self.x.reshape(1, dims, 1)
        elif len(self.x.shape) == 2:
            dims = len(self.x[0])
            mu = self.x.reshape(len(self.x), dims, 1)
        else:
            raise ValueError("Shape of conditioned values not supported.")

        if len(x.shape) == 1:
            x = x.reshape(1, dims, 1)
        elif len(x.shape) == 2:
            x = x.reshape(len(x), dims, 1)
        else:
            raise ValueError("Shape of samples not supported.")

        cov = np.diag(np.ones(dims)) * self.noise_level
        inv_cov = np.linalg.inv(cov)
        log_det = np.log(np.linalg.det(cov))

        term1 = - 0.5 * dims * np.log(np.pi * 2)
        term2 = - 0.5 * log_det

        diff = x - mu
        term3 = - 0.5 * ((np.transpose(diff, axes=(0, 2, 1))) @ inv_cov @ diff)
        return (term1 + term2 + term3).reshape(len(x))

    def prob(self, x):
        return np.exp(self.logprob(x))

    def is_ready(self):
        return True

    def wait_for_ready(self, timeout):
        return

    def draw(self, axis):
        x = np.linspace(self.supp[0], self.supp[1], 100)
        noise = np.random.normal(0, self.noise_level, 100)
        axis.scatter(x, self.model(x) + noise)

    def condition(self, x):
        self.x = x

    def support(self):
        return self.supp

    def marginal(self, dim):
        raise NotImplementedError

    def integral(self, a, b):
        raise NotImplementedError
