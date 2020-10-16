import numpy as np
import sys
from scipy.special import logsumexp


class CMixtureModel:
    def __init__(self, models, weights):
        assert len(models) == len(weights), "len(models)=%d , len(weights)=%d" % (len(models), len(weights))
        self.models = models
        self.weights = weights
        self.set_weights(np.array(weights))

    def set_weights(self, weights):
        assert len(self.models) == len(weights), "len(models)=%d , len(weights)=%d" % (len(self.models), len(weights))

        # Address NaN or negative
        indices = np.logical_or(np.isnan(weights), weights <= 0)
        if np.any(indices):
            print("CMixtureModel. WARNING! There are NaN, negative or zero weights. Setting their weights to zero", file=sys.stderr)
            weights[indices] = 0

        # Make sure weights are normalized
        if weights.sum() <= 0:
            print("CMixtureModel. ERROR! All weights are set to zero!", file=sys.stderr)
        else:
            weights = weights / weights.sum()
        self.weights = weights

    def log_prob(self, data):
        if len(data.shape) == 1:
            data.reshape(-1, 1)
            llikelihood = np.zeros((len(self.models), 1))

        elif len(data.shape) == 2:
            llikelihood = np.zeros((len(self.models), len(data)))
        else:
            raise ValueError("Unsupported samples data format: " + str(data.shape))

        for i in range(len(self.models)):
            llikelihood[i] = self.models[i].log_prob(data).reshape(-1)

        logprob = logsumexp(llikelihood, b=self.weights.reshape(-1, 1), axis=0)
        return logprob

    def prob(self, data):
        if len(data.shape) == 1:
            data.reshape(-1, 1)
            likelihood = np.zeros((len(self.models), 1))

        elif len(data.shape) == 2:
            likelihood = np.zeros((len(self.models), len(data)))
        else:
            raise ValueError("Unsupported samples data format: " + str(data.shape))

        for i in range(len(self.models)):
            likelihood[i] = np.exp(self.models[i].log_prob(data).flatten()) * self.weights[i]

        likelihood = np.sum(likelihood, axis=0)

        zero_mask = likelihood < np.finfo(likelihood.dtype).min
        likelihood[zero_mask] = np.finfo(likelihood.dtype).min

        return likelihood

    def sample(self, n_samples=1):
        res = np.array([])
        for _ in range(n_samples):
            idx = np.argmax(np.random.multinomial(1, self.weights))  # Select the sampling proposal
            q = self.models[idx]
            x = q.sample(1)  # Sample from the sampling proposal
            res = np.concatenate((res, x)) if res.size else x
        return res
