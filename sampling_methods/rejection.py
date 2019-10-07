import numpy as np
from sampling_methods.base import CSamplingMethod


# TODO: Implement importance sampling  for this case
class CRejectionSampling(CSamplingMethod):
    def __init__(self, space_min, space_max, proposal_dist, scaling=1):
        super(self.__class__, self).__init__(space_min, space_max)
        self.range = space_max - space_min
        self.proposal_dist = proposal_dist
        self.scaling = scaling

    def sample(self, n_samples):
        raise NotImplementedError

    def sample_with_likelihood(self, pdf, n_samples):
        # Generate proposal samples from proposal distribution
        proposals = self.proposal_dist.sample(n_samples)

        # Obtain the sample likelihood for both the proposal and target distributions
        proposals_prob_q = np.exp(self.proposal_dist.log_prob(proposals)) * self.scaling
        proposals_logprob_p = pdf.log_prob(proposals)

        # Compute the acceptance ratio
        acceptance_ratio = np.exp(proposals_logprob_p) / proposals_prob_q

        # Generate acceptance probability values from U[0,1)
        acceptance_prob = np.random.uniform(0,1, size=n_samples)

        # Obtain the indices of the accepted samples
        accept_idx = acceptance_ratio > acceptance_prob

        # Return the accepted samples
        return proposals[accept_idx], proposals_logprob_p[accept_idx]
