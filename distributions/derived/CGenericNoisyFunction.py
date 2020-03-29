import numpy as np
from distributions.distributions import CDistribution


class GenericNoisyFunction(CDistribution):
    def __init__(self, params):
        self._check_param(params, "noise_model", CDistribution)
        self._check_param(params, "function")

        params["type"] = "generic"
        params["family"] = "stochastic_function"
        params["likelihood_f"] = self.prob
        params["loglikelihood_f"] = self.log_prob

        super(GenericNoisyFunction, self).__init__(params)

        self.noise_model = params["noise_model"]
        self.function = params["function"]

        assert callable(self.function), "Function must be callable"

        self.x = None

    def sample(self):
        noise = self.noise_model.sample(len(self.x))
        fx = self.function(self.x) + noise
        return fx

    def log_prob(self, x):
        if self.x is None:
            raise ValueError("GenericFunction logprob cannot be evaluated w/o conditioning first with the true x value.")
        return self.noise_model.log_prob(x-self.x)

    def prob(self, x):
        if self.x is None:
            raise ValueError("GenericFunction prob cannot be evaluated w/o conditioning first with the true x value.")

        return self.noise_model.prob(x-self.x)

    def is_ready(self):
        return True

    def wait_for_ready(self, timeout):
        return

    def draw(self, axis, n_points=100, label=None, color=None):
        # TODO: Check dimensions and make plot for 2d as well

        z = np.linspace(self.support_vals[0], self.support_vals[1], n_points).reshape(n_points, 1)
        self.condition(z)
        samples = self.sample()
        axis.scatter(z, samples, c=color, label=label)

        xs = self.sample()
        for _ in range(200):
            x = self.sample()
            xs = np.dstack((xs, x))

        means = np.mean(xs, axis=2).flatten()
        stdevs = np.std(xs, axis=2).flatten()
        if label is not None:
            axis.fill_between(z.flatten(), means - 3 * stdevs, means + 3 * stdevs, label=label + " $3\sigma$", color=color, alpha=0.5)
            axis.plot(z.flatten(), means, label=label + " mean", color=color, alpha=0.5)
        else:
            axis.fill_between(z.flatten(), means - 3 * stdevs, means + 3 * stdevs, color=color, alpha=0.5)
            axis.plot(z.flatten(), means, color=color)

    def condition(self, x):
        self.x = x

    def marginal(self, dim):
        raise NotImplementedError

    def integral(self, a, b):
        raise NotImplementedError


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from distributions.parametric.CMultivariateNormal import CMultivariateNormal

    mean = 0
    sigma = 0.01

    params = dict()
    params["noise_model"] = CMultivariateNormal({"mean": np.array([mean]), "sigma": np.diag([sigma])})
    params["function"] = lambda x: 1 / (1+np.e**-x)
    params["support"] = [-6, 6]
    params["dims"] = 1
    dist = GenericNoisyFunction(params)

    # Condition the generative process on a value and evaluate the likelihoods of the noisy process
    x = 0.6
    dist.condition(np.array([x]))
    px = dist.prob(np.array([[x], [x + sigma], [x - sigma]]))
    lpx = dist.log_prob(np.array([[x], [x + sigma], [x - sigma]]))

    print("x = ", x)
    print("p(x) = ", px)
    print("log(p(x)) = ", lpx)

    plt.figure()
    dist.draw(plt.gca())
    plt.show(True)
