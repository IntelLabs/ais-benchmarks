import numpy as np
import time
from sampling_methods.base import CSamplingMethod
from sampling_methods.base import t_tensor
from utils.plot_utils import plot_pdf
from distributions.CMultivariateNormal import CMultivariateNormal

class CMetropolisHastings(CSamplingMethod):
    """
    Class implementing metropolis-hastings algorithm. Usages:
        - Generating samples from an unknown distribution which likelihood can be evaluated.
        - Integral computations of high dimensional functions

    PARAMETERS:
    n_steps: number of steps between samples. This can be used to reduce the correlation of subsequent samples.

    n_burnin: number of samples considered as burn-in. This is the number of MH steps that it takes to the MCMChain
                to converge to the stationary distribution where samples can be produced

    proposal_d: Proposal distribution used to propose the MC moves during the Markov Chain execution, i.e. P(x'|x).
                must implement the .sample() method. Proposals are computed as x' = x + proposal_d.sample()

    target_d: Interface to the target distribution. Must implement the prob(x) and logprob(x) methods in order for the
                algorithm to be able to compute the acceptance ratios.
    """
    REJECT = 0
    ACCEPT = 1
    BURN_IN = 2
    DECORRELATION = 3
    SAMPLE = 4

    def __init__(self, space_min, space_max, params):
        super(self.__class__, self).__init__(space_min, space_max)
        self.proposal_d = params["proposal_d"]
        self.n_steps = params["n_steps"]
        self.n_burnin = params["n_burnin"]
        self.target_d = params["target_d"]
        self.bw = np.array([params["kde_bw"]])
        self.current_sample = self.proposal_d.sample()
        self.is_init = False
        self.trajectory_samples = []
        self.trajectory_types = []  # 0:reject 1:accept 2:burn_in 3:decorrelation 4:sample

    def sample(self, n_samples, timeout=60):
        """
        Generate n_samples from the target distribution
        :param n_samples:
        :param timeout:
        :return: generated samples
        """
        # Run the burn in period if necessary
        if not self.is_init:
            for _ in range(self.n_burnin):
                self.current_sample = self.mcmc_step(self.current_sample)
                self.trajectory_types[-1] = self.BURN_IN
            self.is_init = True

        res = t_tensor([])
        for _ in range(n_samples):
            sample = self.mcmc(n_samples, self.n_steps, timeout)
            res = np.concatenate((res, sample)) if res.size else sample
        return res

    def reset(self):
        super(self.__class__, self).reset()
        self.is_init = False
        self.trajectory_samples = []
        self.trajectory_types = []

    def mcmc_step(self, x_old):
        accepted = False
        while not accepted:
            # Sample from the proposal distribution to obtain the proposed sample x' ~ p(x'|x)
            x_new = x_old + self.proposal_d.sample()
            self._num_q_samples += 1
            self.trajectory_samples.append(x_new)

            # Compute the symmetric acceptance ratio P(x')/P(x). In its log form: log(P(x')) - log(P(x))
            pi_x_new = self.target_d.logprob(x_new)
            pi_x_old = self.target_d.logprob(x_old)
            self._num_pi_evals += 2
            metropolis_term = pi_x_new - pi_x_old

            # Compute the assymetric term Q(x'|x)/Q(x|x'). If the proposal distribution is symmetric
            # Q(x'|x) = Q(x|x') and the assymetric term is 1. In its log form: log(Q(x'|x)) - log(Q(x|x'))
            hastings_term = self.proposal_d.logprob(x_old - x_new) \
                            - self.proposal_d.logprob(x_new - x_old)
            self._num_q_evals += 2

            # Obtain the acceptance ratio. Notice we are working with logs, therefore the sum instead of the product
            alpha = metropolis_term + hastings_term

            # Determine the accept probability a ~ U(0,1)
            accept = np.random.rand()

            # Accept or reject the proposed new sample x'
            accepted = alpha > np.log(accept)

            self.trajectory_types.append(self.REJECT)

        x_old = x_new
        self.trajectory_types.append(self.ACCEPT)
        return x_old

    # TODO: Implement timeout
    def mcmc(self, n_samples, n_steps, timeout=60):
        samples = t_tensor([])
        for _ in range(n_samples):
            for _ in range(n_steps):
                self.current_sample = self.mcmc_step(self.current_sample)
                self.trajectory_types[-1] = self.DECORRELATION
            self.trajectory_types[-1] = self.SAMPLE
            samples = np.concatenate((samples, self.current_sample)) if samples.size else self.current_sample
        return samples

    def importance_sample(self, target_d, n_samples, timeout=60):
        elapsed_time = 0
        t_ini = time.time()

        while n_samples > len(self.samples) and elapsed_time < timeout:
            new_samples = self.sample(1, timeout)
            self.samples = np.concatenate((self.samples, new_samples)) if self.samples.size else new_samples
            elapsed_time = time.time() - t_ini

        return self.samples, np.zeros(len(self.samples))

    def prob(self, s):
        prob = np.zeros_like(s.flatten())
        for x in self.samples:
            q = CMultivariateNormal(x, np.diag(self.bw))
            pval = q.prob(s)
            prob = prob + pval.flatten()
            self._num_q_evals += 1
        return prob / len(self.samples)

    def draw(self, ax):
        res = []
        for sample, type in zip(self.trajectory_samples, self.trajectory_types):
            style = "rX"
            y = 0
            if type == self.BURN_IN:
                style = "ro"
                y = 0.1
            elif type == self.REJECT:
                style = "r."
                y = 0.2
            elif type == self.DECORRELATION:
                style = "go"
                y = 0.3
            elif type == self.SAMPLE:
                style = "gx"
                y = 0.4

            res.extend(ax.plot(sample, y, style))

        res.extend(plot_pdf(ax, self, self.space_min, self.space_max, alpha=1.0, options="r-", resolution=0.01,label="$q(x)$"))
        res.extend(ax.plot(0, 0, "ro", label="burn-in"))
        res.extend(ax.plot(0, 0, "r.", label="rejected"))
        res.extend(ax.plot(0, 0, "go", label="intermediate"))
        res.extend(ax.plot(0, 0, "gx", label="sample"))
        return res
