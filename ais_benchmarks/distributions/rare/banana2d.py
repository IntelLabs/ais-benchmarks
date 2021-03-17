import numpy as np
from distributions.distributions import CDistribution


class Banana2D(CDistribution):
    """
    A density used by Lu et al. for benchmarking HiDaisee. [Lu, Xiaoyu, Tom Rainforth, Yuan Zhou, Jan-Willem van de
    Meent, and Yee Whye Teh. “On Exploration, Exploitation and Learning in Adaptive Importance Sampling.”
    ArXiv:1810.13296 [Cs, Stat], October 31, 2018. http://arxiv.org/abs/1810.13296.

    f(x1 , x2) \\propto exp{−0.5(0.03*x_1*x_1 + (x_2 + 0.03(x_1*x_1 − 100))^2)}
    """

    def __init__(self, params):
        """
        :param params: A dictionary containing the distribution parameters. Must define the following parameters:
                params["dims"]
        """
        params["dims"] = 2
        params["family"] = "exponential"
        params["type"] = "banana"
        params["likelihood_f"] = self.prob
        params["loglikelihood_f"] = self.log_prob
        params["mean"] = np.array([1])
        params["support"] = np.array([[-10, -10], [10, 10]])
        super(self.__class__, self).__init__(params)

    def log_prob(self, samples):
        if len(samples.shape) == 1:
            samples = samples.reshape(1, self.dims, 1)
        elif len(samples.shape) == 2:
            samples = samples.reshape(len(samples), self.dims, 1)
        else:
            raise ValueError("Shape of samples does not match self.dims")

        return self.logdensity(samples[:, 0, :], samples[:, 1, :]).reshape(len(samples))

    def logdensity(self, x1, x2):
        return -0.5 * (0.03 * x1 * x1 + (x2 + 0.03 * (x1 * x1 - 100))**2)

    def prob(self, samples):
        return np.exp(self.log_prob(samples))

    def condition(self, dist):
        raise NotImplementedError

    def marginal(self, dim):
        raise NotImplementedError


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    mean = np.array([1.])
    dist = Banana2D(dict())

    plt.figure()
    plt.title('Banana2D')
    dist.draw(plt.gca())
    plt.show()
