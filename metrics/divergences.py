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

    def pre(self, **kwargs):
        pass

    def post(self, **kwargs):
        pass

    def reset(self, **kwargs):
        pass

    def compute(self, **kwargs):
        assert "p" in kwargs.keys() and hasattr(kwargs["p"], "log_prob"), \
            "p must implement a log_prob method. p is of type %s" % (type(kwargs["p"]))
        assert "q" in kwargs.keys() and hasattr(kwargs["q"], "log_prob"), \
            "q must implement a log_prob method. q is of type %s" % (type(kwargs["q"]))
        assert "nsamples" in kwargs.keys()

        samples = np.random.uniform(kwargs["p"].support()[0], kwargs["p"].support()[1],
                                    size=(kwargs["nsamples"], kwargs["p"].dims))

        return self.compute_from_samples(kwargs["p"], kwargs["q"], samples)

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
        self.value = np.sum(self.compute_from_log_probs(p_samples_logprob, q_samples_logprob))

        return self.value

    @staticmethod
    def compute_from_probs(p_samples_prob, q_samples_prob):
        # Filter zero prob values on any of the sides
        inliers = np.logical_and(p_samples_prob > 0, q_samples_prob > 0)
        num_inliers = len(p_samples_prob[inliers])
        inlier_ratio = num_inliers / len(p_samples_prob)
        if  inlier_ratio < .1:
            print("WARNING. JSD compute. Less than 10% inliers")

        # Normalize sample probabilities
        if np.sum(p_samples_prob[inliers]) > 0:
            p_samples_prob[inliers] = p_samples_prob[inliers] / np.sum(p_samples_prob[inliers])
        if np.sum(q_samples_prob[inliers]) > 0:
            q_samples_prob[inliers] = q_samples_prob[inliers] / np.sum(q_samples_prob[inliers])
        else:
            return np.inf

        # Compute discrete KL for each sample
        res = p_samples_prob[inliers] * np.log(p_samples_prob[inliers] / q_samples_prob[inliers])
        res[q_samples_prob[inliers] <= 0] = 0
        return res

    @staticmethod
    def compute_from_log_probs(p_samples_logprob, q_samples_logprob):
        # Filter zero prob values on any of the sides
        inliers = np.logical_and(p_samples_logprob > -np.inf, q_samples_logprob > -np.inf)
        num_inliers = len(p_samples_logprob[inliers])
        inlier_ratio = num_inliers / len(p_samples_logprob)

        if  inlier_ratio < .1:
            print("WARNING. KLDD compute. Less than 10% inliers")

        # Normalize sample probabilities in log space
        p_samples_logprob[inliers] -= np.logaddexp.reduce(p_samples_logprob[inliers])
        q_samples_logprob[inliers] -= np.logaddexp.reduce(q_samples_logprob[inliers])

        # Compute discrete KL for each sample
        res = np.exp(p_samples_logprob[inliers]) * (p_samples_logprob[inliers] - q_samples_logprob[inliers])
        res[np.isnan(q_samples_logprob[inliers])] = np.inf
        return res


class CJSDivergence(CDivergence):
    def __init__(self):
        super().__init__()
        self.name = "JSD"
        self.type = "divergence"
        self.is_symmetric = True
        self.disjoint_support = False
        self.log2 = np.log(2)

    def compute_from_samples(self, p, q, samples):
        # Obtain sample probabilities
        p_samples_prob = p.prob(samples)
        q_samples_prob = q.prob(samples)

        # Filter zero prob values on any of the sides
        inliers = np.logical_and(p_samples_prob > 0, q_samples_prob > 0)
        num_inliers = len(p_samples_prob[inliers])
        inlier_ratio = num_inliers / float(len(p_samples_prob))
        if  inlier_ratio < .1:
            print("WARNING. JSD compute. Only %6.3f%% inliers" % (inlier_ratio*100))

        # Normalize sample probabilities
        if np.sum(p_samples_prob[inliers]) > 0:
            p_samples_prob[inliers] = p_samples_prob[inliers] / np.sum(p_samples_prob[inliers])
        if np.sum(q_samples_prob[inliers]) > 0:
            q_samples_prob[inliers] = q_samples_prob[inliers] / np.sum(q_samples_prob[inliers])
        else:
            return np.inf

        m_samples_prob = 0.5 * p_samples_prob[inliers] + 0.5 * q_samples_prob[inliers]
        if np.sum(m_samples_prob) > 0:
            m_samples_prob = m_samples_prob / np.sum(m_samples_prob)

        res = 0.5 * p_samples_prob[inliers] * np.log(p_samples_prob[inliers] / m_samples_prob) + \
              0.5 * q_samples_prob[inliers] * np.log(q_samples_prob[inliers] / m_samples_prob)

        self.value = res.sum() / self.log2
        return res.sum() / self.log2
