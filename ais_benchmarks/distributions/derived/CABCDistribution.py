import numpy as np
from ais_benchmarks.distributions import CDistribution


class ABCDistribution(CDistribution):
    """
    This class represents a probability distribution that is constructed via Approximate Bayesian Computations (ABC).
    ABC is an analysis by synthesis approach where a generative model (i.e "gen_d" parameter) is used to compute the
    posterior distribution of the variables of interest conditioned on observations. Observations are obtained via the
    sensor model (i.e. sensor_d parameter) of the true data generating process. The ranking is done with a surrogate
    likelihood function (i.e "likelihood_f" parameter). The Bayes rule is completed by the prior distribution (i.e.
    "prior_d" parameter). If the prior is unknown a uniform over the domain can be used. The domain must be specified
    via the "support" parameter. The "slack" parameter is used by the ABC likelihood to account for the difference
    between the generative model and the true data generating process.

    TODO: Check why increasing the sensor noise in the example does not increase the uncertainty in the posterior.
    """
    def __init__(self, params):
        """
        :param params: A dictionary containing the distribution parameters. Must define the following parameters:
                params["dims"]
                params["likelihood_f"]
                params["loglikelihood_f"]
                params["support"]
                params["prior_d"]
                params["sensor_d"]
                params["gen_d"]
                params["slack"]
        """
        self._check_param(params, "prior_d", CDistribution)
        self._check_param(params, "sensor_d", CDistribution)
        self._check_param(params, "gen_d", CDistribution)
        self._check_param(params, "likelihood_f")
        self._check_param(params, "loglikelihood_f")
        self._check_param(params, "slack")

        params["type"] = "abc"
        params["family"] = "sampled"
        params["dims"] = params["gen_d"].dims
        super(ABCDistribution, self).__init__(params)

        self.set_likelihood_f(params["likelihood_f"])
        self.set_loglikelihood_f(params["loglikelihood_f"])

        self.prior_d = params["prior_d"]
        self.sensor_d = params["sensor_d"]
        self.gen_d = params["gen_d"]
        self.slack = params["slack"]
        self.o = None

    def condition(self, o):
        self.o = o

    def generate(self):
        self.gen_d.set_params(z)        # Set the model parameters to the evaluated params
        self.gen_d.condition(self.o)    # Condition the generative model at the points to be evaluated
        fx = self.gen_d.sample()        # Evaluate the generative model at the observed values

        self.sensor_d.condition(fx)     # Condition the sensor model on the generated observation
        o_hat = self.sensor_d.sample()  # Obtain the generated observation from the sensor model
        return o_hat

    def log_prob(self, z):
        if self.o is None:
            raise ValueError("ABCDistribution prob or logprob cannot be evaluated w/o conditioning it first with \
                             an observation. Make sure to call self.condition(obs) first.")

        o_hat = self.generate()
        prior_lprob = self.prior_d.log_prob(z).reshape(len(z), 1)  # Evaluate the prior probability of the latent values
        sensor_lprob = 0.0  # TODO: Consider adding the sensor model probability into de computation
        # sensor_lprob = self.sensor_d.log_prob(o_hat)  # Evaluate the measurement probability of the generated observation

        # Evaluate the ABC posterior using the proxy likelihood function with the generated and the observed
        # values, the prior probability of the parameter values and the measurement probability. Prefer to use log_prob
        # function if provided for numerical stability.
        if self.loglikelihood_f is not None:
            return (self.loglikelihood_f(self.o, o_hat, self.slack) + prior_lprob + sensor_lprob).reshape(len(z), 1)
        elif self.likelihood_f is not None:
            return (np.log(
                self.likelihood_f(self.o, o_hat, self.slack)) + prior_lprob + sensor_lprob).reshape(len(z), 1)
        else:
            raise Exception("ABCDistribution. likelihood_f and loglikelihood_f not defined. Must define at least one \
                             of them.")

    def prob(self, z):
        if self.o is None:
            raise ValueError("ABCDistribution prob or logprob cannot be evaluated w/o conditioning it first with \
                             an observation. Make sure to call self.condition(obs) first.")

        o_hat = self.generate()
        prior_lprob = self.prior_d.log_prob(z).reshape(len(z), 1)    # Evaluate the prior probability of the latent values
        sensor_lprob = 0.0  # TODO: Consider adding the sensor model probability into de computation
        # sensor_lprob = self.sensor_d.log_prob(o_hat)  # Evaluate the measurement probability of the generated observation

        # Evaluate the ABC posterior using the proxy likelihood function with the generated and the observed
        # values, the prior probability of the parameter values and the measurement probability. Prefer to use log_prob
        # function if provided for numerical stability.
        if self.loglikelihood_f is not None:
            return np.exp(
                self.loglikelihood_f(self.o, o_hat, self.slack) + prior_lprob + sensor_lprob).reshape(len(z), 1)
        elif self.likelihood_f is not None:
            return self.likelihood_f(self.o, o_hat, self.slack).reshape(len(z), 1) * np.exp(prior_lprob + sensor_lprob)
        else:
            raise Exception("ABCDistribution. likelihood_f and loglikelihood_f not defined. Must define at least one \
                             of them.")

    def sample(self, nsamples=1):
        z = self.prior_d.sample(nsamples)

        self.gen_d.condition(z)
        x = self.gen_d.sample(nsamples)

        self.sensor_d.condition(x)
        o_hat = self.sensor_d.sample(nsamples)

        return o_hat

    def draw(self, ax, n_points=100, label=None, color=None):
        # TODO: Check dimensions and make plot for 2d as well
        z = np.linspace(self.support()[0], self.support()[1], n_points)
        # z = self.prior_d.sample(n_points).flatten()
        self.gen_d.set_params(z)

        # For each point in the latent space model, we need to sample the posterior
        xs = self.gen_d.sample()
        self.sensor_d.condition(xs)
        o_hats = self.sensor_d.sample()
        for _ in range(n_points-1):
            x = self.gen_d.sample()
            xs = np.vstack((xs, x))
            self.sensor_d.condition(x)
            o_hat = self.sensor_d.sample()
            o_hats = np.vstack((o_hats, o_hat))

        means = np.mean(o_hats, axis=0)
        stdevs = np.std(o_hats, axis=0)
        ax.fill_between(z, means - 3 * stdevs, means + 3 * stdevs, label=label + " $3\sigma$", color=color, alpha=0.5)
        ax.plot(z, means, label=label, c=color)

    def marginal(self, dim):
        raise NotImplementedError

    def integral(self, a, b, nsamples=None):
        raise NotImplementedError


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from distributions.derived.CGenericNoisyFunction import GenericNoisyFunction
    from distributions.parametric.CMultivariateNormal import CMultivariateNormal

    # Set-up the surrogate likelihood function, this example is just a multivariate normal
    def loglikelihood_f(x, y, slack):
        dist = CMultivariateNormal({"mean": y, "sigma": np.diag(np.ones(len(y)) * slack)})
        logprob = dist.log_prob(x)
        return logprob

    # Set-up the generative model
    gen_params = dict()
    gen_params["noise_model"] = CMultivariateNormal({"mean": np.array([0]), "sigma": np.diag([0.01])})
    gen_params["function"] = lambda z, a=3, f=0.5, phi=0: (a * np.sin(z * 2 * np.pi * f + phi) + a) / 2
    gen_params["support"] = np.array([-1, 1])
    gen_params["dims"] = 1
    gen_model = GenericNoisyFunction(gen_params)

    # Set-up a generic noisy sensor model
    sensor_params = dict()
    sensor_params["noise_model"] = CMultivariateNormal({"mean": np.array([0]), "sigma": np.diag([0.01])})
    sensor_params["function"] = lambda x: x
    sensor_params["support"] = gen_params["support"]
    sensor_params["dims"] = 1
    sensor_model = GenericNoisyFunction(sensor_params)

    # Set-up the ABC distribution
    params = dict()
    params["dims"] = 1
    params["likelihood_f"] = lambda x, y, slack: np.exp(loglikelihood_f(x, y, slack))
    params["loglikelihood_f"] = loglikelihood_f
    params["support"] = [-1, 1]
    params["prior_d"] = CMultivariateNormal({"mean": np.array([0]), "sigma": np.diag([0.2])})
    params["sensor_d"] = sensor_model
    params["gen_d"] = gen_model
    params["slack"] = 0.1
    dist = ABCDistribution(params)

    # Plot the components of the ABC distribution
    plt.figure()
    dist.draw(plt.gca(), label="ABC Distribution")
    params["prior_d"].draw(plt.gca(), label="Prior")

    # Plot the probability conditioned on an observed value
    observation = 0.5
    dist.condition(np.array([observation]))
    plt.axhline(observation, label="Observation", linestyle="--", c="g")

    # Sample it at a fine grained grid. More sophisticated posterior inference techniques can be used, see the
    # sampling_methods examples.
    z = np.linspace(-1, 1, 300).reshape(300, 1)
    px = np.array([])
    for x in z:
        px = np.concatenate((px, dist.prob(np.array([x])).flatten()))

    # Normalize px. The ABC distribution uses a surrogate likelihood that is not guaranteed to be normalized.
    px = px / np.sum(px)

    # Rescale the normalized px values by the grid cell volume [max-min / npoints]
    plt.plot(z.flatten(), px.flatten() / (2.0/float(len(z))), label="Ground truth posterior", color="c")
    plt.fill_between(z.flatten(), px.flatten() / (2.0/float(len(z))), label="Posterior density", color="y", alpha=0.5)

    plt.xlim(gen_params["support"])
    plt.legend()
    plt.show(True)
