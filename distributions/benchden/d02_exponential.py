import numpy as np
from distributions.parametric.CMultivariateExponential import CMultivariateExponential


class BenchDenExponential(CMultivariateExponential):
    """
    A density part of the benchmark described in [Mildenberger, Thoralf, and Henrike Weinert. “The Benchden Package:
    Benchmark Densities for Nonparametric Density Estimation.” Journal of Statistical Software 46, no. 1 (March 7,
    2012): 1–14. https://doi.org/10.18637/jss.v046.i14.]

    2. Exponential: The density of the Exp(1) exponential distribution.
    """

    def __init__(self, params):
        """
        :param params: A dictionary containing the distribution parameters. Must define the following parameters:
                params["dims"]
        """
        params["dims"] = 1
        params["mean"] = np.array([1])
        params["support"] = np.array([0, 4])
        super(self.__class__, self).__init__(params)
