import time
import numpy as np
from abc import ABCMeta, abstractmethod
from utils.plot_utils import plot_pdf2d
from utils.plot_utils import plot_pdf


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
             over the support subspace is 1. support[0] contains the lower limits and support[1] the upper limits for
             the support of the distribution. The support limits are used to generate the samples for the approximate
             JSD and other MC approximations for the metrics. If the target has infinite support, a reasonable support
             containing most of the probability mass needs to be specified.

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

    TODO:
        Add other useful abstract methods and propagate to existing implementations and tests
            def prob_grad(self, x)
            def log_prob_grad(self, x)
            def cdf(self, x)
            def log_cdf(self, x)
            def entropy(self)
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

    def _check_shape(self, x):
        """
        :param x: Points to evaluate the PDF. Checks that x is NxM array that contains N points of M dimensions.
                  Where M has to match the dimensionality of the PDF represented by this class instance,
                  i.e. M == self.dims.
        :return: Reshaped version of x that fulfills the dimension criteria or x itself if all the checks are correct.
        :raises: ValueError. In the case that the shape x does not fit the criteria and cannot be reshaped.
        """
        # Add the batch dimension if the shape of the evaluated samples does not have the desired NxM elements.
        if len(x.shape) == 1:
            x = x.reshape(1, self.dims)
        # Check that the sample dimensions matches the PDF.
        elif len(x.shape) == 2:
            if len(x[0]) != self.dims:
                raise ValueError("""Shape of x does not match self.dims = %d. is Nx%d array that contains N points of %d 
                                    dimensions. x.shape=%s""" % (self.dims, self.dims, self.dims, str(x.shape)))
        else:
            raise ValueError("""Shape of x does not match self.dims = %d. is Nx%d array that contains N points of %d 
                                dimensions. x.shape=%s""" % (self.dims, self.dims, self.dims, str(x.shape)))
        return x

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

    def sample(self, nsamples=1):
        """
        Generate nsamples from the distribution represented by this instance. If the derived class does not implement
        this method, it will fall back to the rejection sampling procedure implemented here. It is highly recommended
        to implement your sampling method for each custom distribution to avoid the low sample efficiency of rejection
        sampling.

        :param nsamples: Desired number of samples to generate.
        :return: N by M array with N = nsamples and M = dimensions of the random variable represented by
                 this distribution.
        """
        # TODO: LOW: Implement generic rejection sampling
        # TODO: LOW: Implement generic MCMC sampling
        raise NotImplementedError

    @abstractmethod
    def condition(self, dist):
        """
        Condition the current distribution with a known constraint that must be satisfied represented by the dist
        passed as a parameter. Depending on the type and dimensionality of the distribution the behavior of the
        condition method can change.
        :param dist: Distribution used to condition this distribution.
        :return: None

        TODO: Usage example and utility and examples of different behavior of this function.
        """
        raise NotImplementedError

    @abstractmethod
    def marginal(self, dim):
        """
        Marginalize the desired dimension of the random variable and return the marginalized distribution.
        :param dim: Dimension to marginalize.
        :return: Distribution resulting of marginalizing the desired dimension.

        TODO: Usage example and utility
        """
        raise NotImplementedError

    def integral(self, a, b, nsamples=10000000):
        print("WARNING! Distribution specific integral not implemented for %s. Resorting to Monte-Carlo integration." % self.__class__.__name__)
        samples = np.random.uniform(a, b, size=(nsamples, self.dims))
        probs = self.prob(samples).flatten()
        height = np.max(probs)
        points = np.random.uniform(0, height, nsamples)
        inliers = probs >= points
        inliers_count = np.sum(inliers)
        volume = np.prod(b - a) * height
        return (inliers_count / nsamples) * volume

    def support(self):
        """

        :return: A 2 by N array with the support of the distribution. [lower_bound, upper_bound] Where each component
                 is N dimensional vector with the dimension-wise support boundaries.
        """
        return np.array(self.support_vals).reshape(2, self.dims)

    def draw(self, ax, resolution=.01, label=None, color=None):
        range = self.support()[1] - self.support()[0]
        if self.dims == 1:
            plot_pdf(ax, self,
                     self.support()[0] - range * .1,
                     self.support()[1] + range * .1,
                     resolution, label=label, color=color)
        elif self.dims == 2:
            plot_pdf2d(ax, self,
                       self.support()[0] - range * .1,
                       self.support()[1] + range * .1)
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
