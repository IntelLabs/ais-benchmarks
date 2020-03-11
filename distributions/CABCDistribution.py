import numpy as np
from distributions.base import CDistribution


class ABCDistribution(CDistribution):
    """
    This class represents a probability distribution that is constructed via Approximate Bayesian Computations (ABC).
    ABC is an analysis by synthesis approach where a generative model (i.e "gen_d" parameter) is used to compute the
    posterior distribution of the variables of interest conditioned on observations. Observations are obtained via the
    sensor model (i.e. sensor_d parameter) of the true data generating process. The ranking is done with a surrogate
    likelihood function (i.e "likelihood_f" parameter). The Bayes rule is completed by the prior distribution (i.e.
    "prior_d" parameter). If the prior is unknown a uniform over the domain can be used. The domain must be specified
    via the "support" parameter. The "slack" parameter is used by the ABC likelihood to accommodate the difference
    between the generative model and the true data generating process.
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
        params["type"] = "abc"
        params["family"] = "sampled"
        super(ABCDistribution, self).__init__(params)
        self.prior_d = params["prior_d"]
        self.sensor_d = params["sensor_d"]
        self.gen_d = params["gen_d"]
        self.slack = params["slack"]
        self.o = None

    def condition(self, o):
        self.o = o

    def log_prob(self, z):
        pz = self.prob(z)
        return np.log(pz)

    def prob(self, z):
        if self.o is None:
            raise ValueError("ABCDistribution prob or logprob cannot be evaluated w/o conditioning it first with \
                             an observation. Make sure to call self.condition(obs) first.")

        self.gen_d.condition(z)
        x = self.gen_d.sample()

        # The sensor model is applied w/o noise to compute the prob of a hipothesis z. The noise is in the observation o
        self.sensor_d.condition(x)
        o_hat = self.sensor_d.model(x)

        # return self.likelihood_f(self.o, o_hat, self.slack) * self.sensor_d.prob(o_hat) * self.prior_d.prob(z)
        return self.likelihood_f(self.o, o_hat, self.slack) * self.prior_d.prob(z)

    def sample(self):
        z = self.prior_d.sample()

        self.gen_d.condition(z)
        x = self.gen_d.sample()

        self.sensor_d.condition(x)
        o_hat = self.sensor_d.sample()

        return o_hat

    def draw(self, ax, n_points=100, label=None, color=None):
        z = np.linspace(self.support()[0], self.support()[1], 100)
        self.gen_d.condition(z)
        x = self.gen_d.sample()

        self.sensor_d.condition(x)
        o_hat = self.sensor_d.sample()
        ax.scatter(z, o_hat, label=label, c=color)

    def marginal(self, dim):
        raise NotImplementedError

    def integral(self, a, b):
        raise NotImplementedError
