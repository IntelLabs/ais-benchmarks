import numpy as np
from ais_benchmarks.distributions import CDistribution


class BenchDenLaplace(CDistribution):
    """
    A density part of the benchmark described in [Mildenberger, Thoralf, and Henrike Weinert. “The Benchden Package:
    Benchmark Densities for Nonparametric Density Estimation.” Journal of Statistical Software 46, no. 1 (March 7,
    2012): 1–14. https://doi.org/10.18637/jss.v046.i14.]

    4. Double exponential: The standard double exponential (or Laplace) distribution with density given by
       f (x) = 1/2 exp(−|x|).
    """

    def __init__(self, params):
        """
        :param params: A dictionary containing the distribution parameters. Must define the following parameters:
                params["dims"]
        """
        params["dims"] = 1
        params["family"] = "exponential"
        params["type"] = "laplace"
        params["likelihood_f"] = self.prob
        params["loglikelihood_f"] = self.log_prob
        params["mean"] = np.array([1])
        params["support"] = np.array([-6, 6])
        super(self.__class__, self).__init__(params)

    def log_prob(self, samples):
        if len(samples.shape) == 1:
            samples = samples.reshape(1, self.dims, 1)
        elif len(samples.shape) == 2:
            samples = samples.reshape(len(samples), self.dims, 1)
        else:
            raise ValueError("Shape of samples does not match self.dims")

        res = np.log(.5) - np.fabs(samples)
        res = np.sum(res, axis=1)
        return res

    def prob(self, samples):
        return np.exp(self.log_prob(samples))

    def condition(self, dist):
        raise NotImplementedError

    def marginal(self, dim):
        raise NotImplementedError


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    mean = np.array([1.])
    dist = BenchDenLaplace(dict())

    plt.figure()
    plt.title('BenchDenLaplace')
    dist.draw(plt.gca())
    plt.show()
