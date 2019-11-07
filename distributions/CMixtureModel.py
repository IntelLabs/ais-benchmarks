import numpy as np
import sys


class CMixtureModel:
    def __init__(self, models, weights):
        assert len(models) == len(weights), "len(models)=%d , len(weights)=%d" % (len(models), len(weights))
        self.models = models
        self.weights = weights
        self.set_weights(weights)

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

    # TODO: Is it possible to avoid computing the prob and stay in the log space?? This is numerically not stable for
    #       higher dimensions
    def logprob(self, data):
        return np.log(self.prob(data))

    def prob(self, data):
        likelihood = np.zeros(len(data))
        if len(data.shape) == 1:
            likelihood = 0

        for i in range(len(self.models)):
            likelihood_i = np.exp(self.models[i].logprob(data)) * self.weights[i]
            likelihood = likelihood + likelihood_i

        zero_mask = likelihood < np.finfo(likelihood.dtype).eps
        likelihood[zero_mask] = np.finfo(likelihood.dtype).eps

        return likelihood

    def sample(self, n_samples):
        res = np.array([])
        for _ in range(n_samples):
            idx = np.argmax(np.random.multinomial(1, self.weights))  # Select the sampling proposal
            q = self.models[idx]
            x = q.sample(1)  # Sample from the sampling proposal
            res = np.concatenate((res, x)) if res.size else x
        return res