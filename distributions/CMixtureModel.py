import numpy as np


class CMixtureModel:
    def __init__(self, models, weights):

        assert len(models) == len(weights), "len(models)=%d , len(weights)=%d" % (len(models), len(weights))

        # Make sure all weights are positive
        assert np.all(weights > 0), "There are non-positive weights"

        # Make sure weights are normalized
        weights = weights / weights.sum()

        self.models = models
        self.weights = weights

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