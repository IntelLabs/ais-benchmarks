import numpy as np
from ais_benchmarks.distributions import CDistribution


class BenchDenSymmetricPareto(CDistribution):
    """
    A density part of the benchmark described in [Mildenberger, Thoralf, and Henrike Weinert. “The Benchden Package:
    Benchmark Densities for Nonparametric Density Estimation.” Journal of Statistical Software 46, no. 1 (March 7,
    2012): 1–14. https://doi.org/10.18637/jss.v046.i14.]

    10. Symmetric Pareto: The symmetric Pareto distribution with parameter 3=2. A translated and symmetrized version
    of density 9. The density function is f(x) = (4 (1 + |x|) ^ (3/2)) ^ −1
    """

    def __init__(self, params):
        """
        :param params: A dictionary containing the distribution parameters. Must define the following parameters:
                params["dims"]
        """
        params["dims"] = 1
        params["family"] = "exponential"
        params["type"] = "pareto"
        params["likelihood_f"] = self.prob
        params["loglikelihood_f"] = self.log_prob
        params["mean"] = np.array([0])
        params["support"] = np.array([-20, 20])
        super(self.__class__, self).__init__(params)

    def log_prob(self, samples):
        return np.log(self.prob(samples))

    def prob(self, samples):
        samples = self._check_shape(samples)
        res = 1 / (4 * (1 + np.fabs(samples)) ** (3/2))
        return res.flatten()

    def condition(self, dist):
        raise NotImplementedError

    def marginal(self, dim):
        raise NotImplementedError


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    mean = np.array([1.])
    dist = BenchDenSymmetricPareto(dict())

    plt.figure()
    plt.title('BenchDenSymmetricPareto')
    dist.draw(plt.gca())
    plt.show()
