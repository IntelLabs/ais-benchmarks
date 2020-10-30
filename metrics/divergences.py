import numpy as np
from metrics.base import CMetric


# def bhattacharyya_distance(p_samples_prob, q_samples_prob):
#     res = - np.log(np.sum(np.sqrt(p_samples_prob * q_samples_prob)))
#     return res
#
#
# def kl_divergence_components_logprob(p_samples_logprob, q_samples_logprob):
#     p_samples_prob = np.exp(p_samples_logprob - np.max(p_samples_logprob))
#     q_samples_prob = np.exp(q_samples_logprob - np.max(q_samples_logprob))
#     res = kl_divergence_components(p_samples_prob, q_samples_prob)
#     return res
#
#
# def kl_divergence_components(p_samples_prob, q_samples_prob):
#     p_samples_prob_norm = p_samples_prob / np.sum(p_samples_prob)
#     q_samples_prob_norm = q_samples_prob / np.sum(q_samples_prob)
#
#     # handle zero probabilities
#     inliers_p = np.logical_and(q_samples_prob_norm > 0, p_samples_prob_norm > 0)
#
#     res = p_samples_prob_norm * (np.log(p_samples_prob_norm) - np.log(q_samples_prob_norm))
#     res[np.logical_not(inliers_p)] = 0
#     return res
#
#
# def kl_divergence_logprob(p_samples_logprob, q_samples_logprob):
#     p_samples_prob = np.exp(p_samples_logprob - np.max(p_samples_logprob))
#     q_samples_prob = np.exp(q_samples_logprob - np.max(q_samples_logprob))
#     res = kl_divergence(p_samples_prob, q_samples_prob)
#     return res
#
#
# def kl_divergence(p_samples_prob, q_samples_prob):
#     p_samples_prob_norm = p_samples_prob / np.sum(p_samples_prob)
#     q_samples_prob_norm = q_samples_prob / np.sum(q_samples_prob)
#     res = (p_samples_prob_norm * np.log(p_samples_prob_norm / q_samples_prob_norm)).sum()
#
#     # res = entropy(pk=p_samples_prob, qk=q_samples_prob)
#     return res
#
#
# def js_divergence(p_samples_prob, q_samples_prob):
#     m_samples_prob = 0.5 * (p_samples_prob + q_samples_prob)
#     res = 0.5 * kl_divergence_components(p_samples_prob, m_samples_prob) + 0.5 * kl_divergence_components(q_samples_prob, m_samples_prob)
#     return res
#
#
# def js_divergence_logprob(p_samples_logprob, q_samples_logprob):
#     p_samples_prob = np.exp(p_samples_logprob - np.max(p_samples_logprob))
#     q_samples_prob = np.exp(q_samples_logprob - np.max(q_samples_logprob))
#     m_samples_prob = 0.5 * (p_samples_prob + q_samples_prob)
#     res = 0.5 * kl_divergence_components(p_samples_prob, m_samples_prob) + \
#           0.5 * kl_divergence_components(q_samples_prob, m_samples_prob)
#
#     return res


class CDivergence(CMetric):
    def __init__(self):
        super().__init__()
        self.name = "base"
        self.type = "divergence"
        self.is_symmetric = False
        self.disjoint_support = True

    def compute(self, params):
        assert "p" in params.keys() and hasattr(params["p"], "log_prob"), \
            "p must implement a log_prob method. p is of type %s" % (type(params["p"]))
        assert "q" in params.keys() and hasattr(params["q"], "log_prob"), \
            "q must implement a log_prob method. q is of type %s" % (type(params["q"]))
        assert "nsamples" in params.keys()

        samples = np.random.uniform(params["p"].support()[0], params["p"].support()[1],
                                    size=(params["nsamples"], params["p"].dims))

        return self.compute_from_samples(params["p"], params["q"], samples)

    def compute_from_samples(self, p, q, samples):
        raise NotImplementedError


class CKLDivergence(CDivergence):
    def __init__(self):
        super().__init__()
        self.name = "KL"
        self.type = "divergence"
        self.is_symmetric = False
        self.disjoint_support = False

    def compute_from_samples(self, p, q, samples):
        # Obtain sample log probabilities
        p_samples_logprob = p.log_prob(samples)
        q_samples_logprob = q.log_prob(samples)

        # Normalize sample probabilities in log space
        p_samples_logprob_norm = p_samples_logprob - np.max(p_samples_logprob)
        q_samples_logprob_norm = q_samples_logprob - np.max(q_samples_logprob)

        # Compute discrete KL for each sample
        res = np.exp(p_samples_logprob_norm) * (p_samples_logprob_norm - q_samples_logprob_norm)
        res[res <= 0] = 0
        return res.sum()
