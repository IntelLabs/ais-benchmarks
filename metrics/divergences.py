"""
TODO: This metrics have to compare the quality of samples that are generated. Therefore we need to consider comparing
        the samples obtained by approximate sampling methods with the target distribution via comparing:
    - distribution vs. distribution
    - samples vs. distribution
    - distribution vs. samples
    - samples vs. samples


TODO: Probability density comparison
    See this fantastic tutorial: http://www.gatsby.ucl.ac.uk/~gretton/papers/neurips19_1.pdf
    - Integral metrics:
        - MMD: Maximum mean discrepancy
        - WD1: Wasserstein Distance - 1
    - f-divergences:
        - Kullback-Liebler Divergence
        - Reverse Kullback-Liebler Divergence
        - Jensen-Shannon Divergence
        - Hellinger Distance
        - Pearson chi^2
    - Total variation

TODO: Some efficiency measurements would also be interesting to compare numpy implementation and scipy of the
      cross entropy. Things that need to be tested/handled:
        - Symmetry of the metric
        - Disjoint support cases
"""

import numpy as np
from metrics.base import CMetric
from scipy.stats import entropy


def bhattacharyya_distance(p_samples_prob, q_samples_prob):
    res = - np.log(np.sum(np.sqrt(p_samples_prob * q_samples_prob)))
    return res


def kl_divergence_components_logprob(p_samples_logprob, q_samples_logprob):
    p_samples_prob = np.exp(p_samples_logprob - np.max(p_samples_logprob))
    q_samples_prob = np.exp(q_samples_logprob - np.max(q_samples_logprob))
    res = kl_divergence_components(p_samples_prob, q_samples_prob)
    return res


def kl_divergence_components(p_samples_prob, q_samples_prob):
    p_samples_prob_norm = p_samples_prob / np.sum(p_samples_prob)
    q_samples_prob_norm = q_samples_prob / np.sum(q_samples_prob)

    # handle zero probabilities
    inliers_p = np.logical_and(q_samples_prob_norm > 0, p_samples_prob_norm > 0)

    res = p_samples_prob_norm * (np.log(p_samples_prob_norm) - np.log(q_samples_prob_norm))
    res[np.logical_not(inliers_p)] = 0
    return res


def kl_divergence_logprob(p_samples_logprob, q_samples_logprob):
    p_samples_prob = np.exp(p_samples_logprob - np.max(p_samples_logprob))
    q_samples_prob = np.exp(q_samples_logprob - np.max(q_samples_logprob))
    res = kl_divergence(p_samples_prob, q_samples_prob)
    return res


def kl_divergence(p_samples_prob, q_samples_prob):
    p_samples_prob_norm = p_samples_prob / np.sum(p_samples_prob)
    q_samples_prob_norm = q_samples_prob / np.sum(q_samples_prob)
    res = (p_samples_prob_norm * np.log(p_samples_prob_norm / q_samples_prob_norm)).sum()

    # res = entropy(pk=p_samples_prob, qk=q_samples_prob)
    return res


def js_divergence(p_samples_prob, q_samples_prob):
    m_samples_prob = 0.5 * (p_samples_prob + q_samples_prob)
    res = 0.5 * kl_divergence_components(p_samples_prob, m_samples_prob) + 0.5 * kl_divergence_components(q_samples_prob, m_samples_prob)
    return res


def js_divergence_logprob(p_samples_logprob, q_samples_logprob):
    p_samples_prob = np.exp(p_samples_logprob - np.max(p_samples_logprob))
    q_samples_prob = np.exp(q_samples_logprob - np.max(q_samples_logprob))
    m_samples_prob = 0.5 * (p_samples_prob + q_samples_prob)
    res = 0.5 * kl_divergence_components(p_samples_prob, m_samples_prob) + \
          0.5 * kl_divergence_components(q_samples_prob, m_samples_prob)

    return res


class CDivergence(CMetric):
    def __init__(self):
        super().__init__()
        self.name = "base"
        self.type = "divergence"
        self.is_symmetric = False
        self.disjoint_support = True

    def compute(self, params):
        assert "p" in params.keys() and type(params["p"]) == CDivergence
        assert "q" in params.keys() and type(params["q"]) == CDivergence
        assert "nsamples" in params.keys()

        samples = np.random.uniform(params["p"].support[0], params["p"].support[1],
                                    size=(params["nsamples"], len(params["p"].dims)))

        return self.compute_from_samples(params["p"], params["q"], samples)

    def compute_from_samples(self, p, q, samples):
        raise NotImplementedError
