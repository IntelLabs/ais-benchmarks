"""
Samples are typically generated from a RV distribution in order to compute some quantities that are useful such
as marginals, confidence intervals, probability of events or expected value. By measuring the error on this statistics
by using the approximated samples and the ground truth values, the quality of sampling algorithms can be assessed.
"""
import numpy as np
from ais_benchmarks.metrics.base import CMetric


class CExpectedValueMSE(CMetric):
    def __init__(self):
        super().__init__()
        self.name = "EVMSE"
        self.type = "statistics"

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
        samples_pprob = p.prob(samples)
        samples_qprob = q.prob(samples)
        samples_pprob /= np.sum(samples_pprob)
        samples_qprob /= np.sum(samples_qprob)

        EV_P = np.sum(samples * samples_pprob.reshape(-1,1), axis=0)
        EV_Q = np.sum(samples * samples_qprob.reshape(-1,1), axis=0)
        res = (EV_P - EV_Q) @ np.transpose(EV_P - EV_Q)
        self.value = res
        return res

