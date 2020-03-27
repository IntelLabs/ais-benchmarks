import time
import numpy as np
from abc import ABCMeta, abstractmethod
from utils.plot_utils import plot_pdf2d


class CDistribution(metaclass=ABCMeta):
    """
    Base class for random variable distributions. Implements generic methods for drawing samples from the RV
    distribution. This base class can be used to model any kind of function that has a stochastic behavior.

    A distribution is composed at its core by:
    1- The sampling function. Generates data points that are distributed as the represented distribution. x ~ P(x)
    2- The likelihood function. Provides the probability density of a random variable value under the
       represented distribution. Accessible through the prob() and log_prob() methods.

    Parameters and properties:
    The distribution class is initialized by a dictionary of parameters which must contain (besides class specific
    parameters) at least the following parameters:
    type: String with the type of distribution. E.g. GMM, Uniform, Beta, Normal, KDE
    family: String with the family of distributions this instance belongs to, e.g. exponential, non-parametric, mixture
    dims: Dimensionality of the random variable.
    support: PDF support. Namely, subspace that contains all the probability density. The definite integral of the PDF
             over the support subspace is 1.
    likelihood_f: Likelihood function. p(x|z) where z are the parameters that define the distribution.
    loglikelihood_f: Log Likelihood function log(p(x|z)). Because probabilities can be very small numbers,
                     log likelihoods are used to avoid numerical stability issues. Besides, sometimes the log form
                     of the likelihood function is easier to compute.

    Batching:
    All methods assume the first dimension to be the batch dimension to support vectorized implementations. The results
    are also in batch form even if there is only one component in the batch dimension.

    Example 1:
    Generative models can be implemented as a distribution base class. A data generating process can be implemented via
    the sample class, that generates a data point from the generative process. If the generative process can be tuned
    depending on some parameters, they can be included via the "condition(dist)" method, that can be used to set the
    internal model parameters that will tune the samples generated.

    If the process is deterministic, the samples generated will always be the same, and the probability of the sample
    will be 1 provided that the same parameters for conditioning have been used.

    Example 2:
    An example of generative models are sensor models which can be implemented by conditioning the distribution on
    the true value "x" and implementing a sample method that models the sensor behavior given the known true state
    "o ~ p(o|x)" that can be used to simulate an observation from the sensor.
    """

    def __init__(self, params):
        """
        :param params: The distribution class is initialized by a dictionary of parameters which must contain (besides
        class specific parameters) at least the following parameters:
        - type String with the type of distribution. E.g. GMM, Uniform, Beta, Normal, KDE
        - family: String with the family of distributions this instance belongs to, e.g. exponential, mixture.
        - dims: Dimensionality of the random variable.
        - support: PDF support. Namely, subspace containing all the probability density. The definite integral of the
                   PDF over the support subspace is 1.
                   WARNING!!: Infinite support distributions are not supported yet. TODO. Must add support for such feature.
        - likelihood_f: Callable likelihood function. p(x|z) where z are the parameters that define the distribution.
                        WARNING: It is assumed that the function passed must support batch evaluations. This means, if
                        the dimensionality of the RV is D, samples to evaluate are passed in a N by D shape where each
                        N is the number of samples, the result must be a N by 1 array with the likelihood results.
        - loglikelihood_f: Callable log Likelihood function log(p(x|z)). Because probabilities can be very small values.
                           log likelihoods are used to avoid numerical stability issues. Besides, sometimes the log form
                           of the likelihood function is easier to compute.
                           WARNING: It is assumed that the function passed must support batch evaluations. This means,
                           if the dimensionality of the RV is D, samples to evaluate are passed in a N by D shape where
                           each N is the number of samples, the result must be a N by 1 array with the likelihood
                           results.
        """
        self._check_param(params, "family")
        self._check_param(params, "dims")
        self._check_param(params, "likelihood_f")
        self._check_param(params, "loglikelihood_f")
        self._check_param(params, "support")

        self.params = params
        self.type = params["type"]
        self.family = params["family"]
        self.dims = params["dims"]
        self.likelihood_f = params["likelihood_f"]
        self.loglikelihood_f = params["loglikelihood_f"]
        self.support_vals = params["support"]

    def _check_param(self, params, param, type=None):
        """
        Private function to assert that input parameters exist and optionally are of de desired type.
        :param params: Parameter dictionary.
        :param param: Name of the parameter to check.
        :param type: Desired type of the parameter.
        :return: None
        """
        assert param in params, "%s '%s' parameter is required" % (self.__class__.__name__, param)
        if type is not None:
            assert isinstance(params[param], type), "%s: Parameter '%s' is required to have type %s" % (self.__class__.__name__, param, str(type))

    def set_likelihood_f(self, likelihood_f):
        """
        Sets the likelihood function. This will be the one used to compute both log_likelihood and likelihood.

        :param likelihood_f: Callable likelihood function. p(x|z) where z are the parameters that define the
                             distribution.
                             WARNING: It is assumed that the function passed must support batch evaluations. This means,
                             if the dimensionality of the RV is D, samples to evaluate are passed in a N by D shape
                             where each N is the number of samples, the result must be a N by 1 array with the
                             likelihood results.
        :return: None
        """
        self.likelihood_f = likelihood_f

    def set_loglikelihood_f(self, loglikelihood_f):
        """
        Sets the log_likelihood function. This will be the one used to compute both log_likelihood and likelihood.

        :param loglikelihood_f: Callable log Likelihood function log(p(x|z)). Because probabilities can be very small
                                values. log likelihoods are used to avoid numerical stability issues. Besides,
                                sometimes the log form of the likelihood function is easier to compute.
                                WARNING: It is assumed that the function passed must support batch evaluations. This
                                means, if the dimensionality of the RV is D, samples to evaluate are passed in a N by D
                                shape where each N is the number of samples, the result must be a N by 1 array with
                                the likelihood results.
        :return: None
        """
        self.loglikelihood_f = loglikelihood_f

    @abstractmethod
    def sample(self, nsamples=1):
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
        if self.dims == 1:
            x = np.linspace(self.support()[0], self.support()[1], n_points).reshape(n_points, 1)
            ax.plot(x.flatten(), self.prob(x).flatten(), label=label, c=color)
        elif self.dims == 2:
            plot_pdf2d(ax, self, self.support()[0], self.support()[1])
        else:
            raise NotImplementedError("Drawing of more than 2D PDF is not implemented in the base CDistribution class. \
                                       Try marginalizing dimensions")

    def prob(self, x):
        """
        Return the PDF evaluated at x.
        :param x: Points to evaluate the PDF. NxM array that contains N points of M dimensions. Where M has to match
        the dimensionality of the PDF represented by this class instance, i.e. M == self.dims.
        :return: Probability density of x.
        """
        assert len(x.shape) == 2 and x.shape[1] == self.dims, "Shape mismatch. x must be an Nx%d array. That contains \
                                                               N points to be evaluated." % self.dims

        if self.likelihood_f is not None:
            return self.likelihood_f(x)
        elif self.loglikelihood_f is not None:
            return np.exp(self.loglikelihood_f(x))
        else:
            raise Exception("Likelihood and LogLikelihood functions not defined")

    def log_prob(self, x):
        """
        Return the log(PDF) evaluated at x.
        :param x: Points to evaluate the PDF. NxM array that contains N points of M dimensions. Where M has to match
        the dimensionality of the PDF represented by this class instance, i.e. M == self.dims.
        :return: Log probability density of x.
        """
        assert len(x.shape) == 2 and x.shape[1] == self.dims, "Shape mismatch. x must be an Nx%d array. That contains \
                                                               N points to be evaluated." % self.dims

        if self.loglikelihood_f is not None:
            return self.loglikelihood_f(x)
        elif self.likelihood_f is not None:
            return np.log(self.likelihood_f(x))
        else:
            raise Exception("Likelihood and LogLikelihood functions not defined")

    def is_ready(self):
        return True

    def wait_for_ready(self, timeout):
        t_ini = time.time()
        while time.time()-t_ini < timeout:
            if self.is_ready():
                return True
            time.sleep(1e-6)
        raise TimeoutError("CDistribution: wait_for_ready timed out.")


class CKernel:
    def __init__(self, loc, bw, func):
        assert callable(func)
        self.kernel = func
        self.loc = loc
        self.bw = bw

    def log_prob(self, x):
        return np.log(self.prob(x))

    def prob(self, x):
        return self(x - self.loc)

    def __call__(self, x):
        return self.kernel((x - self.loc) * self.bw) * self.bw

    @staticmethod
    def kernel_normal(u):
        return 0.3989422804 * np.e ** (-0.5 * u * u)

    @staticmethod
    def kernel_uniform(u):
        res = np.ones_like(u) * 0.5
        mask = np.logical_or(u < -1, u > 1)
        res[mask] = 0
        return res

    @staticmethod
    def kernel_triangular(u):
        res = 1 - np.abs(u)
        mask = np.logical_or(u < -1, u > 1)
        res[mask] = 0
        return res

    @staticmethod
    def kernel_epanechnikov(u):
        res = .75 * (1 - u * u)
        mask = np.logical_or(u < -1, u > 1)
        res[mask] = 0
        return res
