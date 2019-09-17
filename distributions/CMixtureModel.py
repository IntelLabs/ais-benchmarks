import numpy as np


class CMixtureModel:
    def __init__(self, models, weights):

        assert len(models) == len(weights), "len(models)=%d , len(weights)=%d" % (len(models), len(weights))

        # Make sure all weights are positive
        assert np.sum(weights < 0) <= 0, "There are non-positive weights"

        # Make sure weights are normalized
        weights = weights / weights.sum()

        self.models = models
        self.weights = weights

    def log_prob(self, data):
        return np.log(self.prob(data))

    def prob(self, data):
        likelihood = np.zeros(len(data))
        if len(data.shape) == 1:
            likelihood = 0

        for i in range( len(self.models) ):
            likelihood = likelihood + np.exp(self.models[i].log_prob(data)) * self.weights[i]

        zero_mask = likelihood < np.finfo(likelihood.dtype).eps
        likelihood[zero_mask] = np.finfo(likelihood.dtype).eps

        return likelihood
