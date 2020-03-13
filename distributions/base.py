import time
import numpy as np
from abc import ABCMeta, abstractmethod


class CDistribution(metaclass=ABCMeta):
    def __init__(self, params):
        self.params = params
        self.type = params["type"]
        self.family = params["family"]
        self.dims = params["dims"]
        self.likelihood_f = params["likelihood_f"]
        self.loglikelihood_f = params["loglikelihood_f"]
        self.support_vals = params["support"]

    def set_likelihood_f(self, likelihood_f):
        self.likelihood_f = likelihood_f
        self.loglikelihood_f = None

    def set_loglikelihood_f(self, loglikelihood_f):
        self.likelihood_f = None
        self.loglikelihood_f = loglikelihood_f

    @abstractmethod
    def sample(self):
        raise NotImplementedError

    @abstractmethod
    def condition(self, dist):
        raise NotImplementedError

    @abstractmethod
    def marginal(self, dim):
        raise NotImplementedError

    @abstractmethod
    def integral(self, a, b):
        raise NotImplementedError

    def support(self):
        return self.support_vals

    def draw(self, ax, n_points=100, label=None, color=None):
        if self.dims > 1:
            raise NotImplementedError("Drawing of more than 1D PDF is not implemented in the base CDistribution class")
        x = np.linspace(self.support()[0], self.support()[1], n_points).reshape(n_points, self.dims)
        ax.plot(x, self.prob(x), label=label, c=color)

    def prob(self, x):
        if self.likelihood_f is not None:
            return self.likelihood_f(x)
        return np.exp(self.loglikelihood_f(x))

    def log_prob(self, x):
        if self.loglikelihood_f is not None:
            return self.loglikelihood_f(x)
        return np.log(self.likelihood_f(x))

    def is_ready(self):
        return True

    def wait_for_ready(self, timeout):
        t_ini = time.time()
        while time.time()-t_ini < timeout:
            if self.is_ready():
                return True
            time.sleep(1e-6)
        raise TimeoutError("CDistribution: wait_for_ready timed out.")
