import numpy as np
from sampling_methods.base import CSamplingMethod


class CMetropolisHastings(CSamplingMethod):
    def __init__(self, space_min, space_max, proposal_dist):
        super(self.__class__, self).__init__(space_min, space_max)
        self.proposal = proposal_dist
        self.current_sample = self.proposal.sample()

    def sample(self, n_samples):
        raise NotImplementedError

    def sample_with_likelihood(self, pdf, n_samples):
        samples = self.current_sample
        values = pdf.log_prob(self.current_sample)
        while len(values) < n_samples:

            # Sample from the proposal distribution to obtain the proposed next sample x'
            proposal_sample = self.proposal.sample() + self.current_sample

            # Compute the symmetric acceptance ratio P(x')/P(x). In its log form: log(P(x')) - log(P(x))
            proposal_sample_logprob = pdf.log_prob(proposal_sample)
            metropolis_term = proposal_sample_logprob - pdf.log_prob(self.current_sample)

            # Compute the assymetric term Q(x'|x)/Q(x|x'). If the proposal distribution is symmetric
            # Q(x'|x) = Q(x|x') and the assymetric term is 1. In its log form: log(Q(x'|x)) - log(Q(x|x'))
            hastings_term = self.proposal.log_prob(self.current_sample - proposal_sample) - \
                            self.proposal.log_prob(proposal_sample - self.current_sample)

            # Obtain the acceptance ratio. Note we are working with logs, therefore the sum instead of the product
            alpha = metropolis_term + hastings_term

            # Determine the accept probability
            accept = np.random.rand()

            # Accept or reject the proposed new sample x'
            if alpha > np.log(accept):
                self.current_sample = proposal_sample
                samples = np.vstack((samples, self.current_sample))
                values = np.concatenate((values, proposal_sample_logprob))

        return samples, values
