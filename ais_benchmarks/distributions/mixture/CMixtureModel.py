import numpy as np
import sys
from scipy.special import logsumexp
from ais_benchmarks.distributions import CDistribution


class CMixtureModel(CDistribution):
    def __init__(self, params):
        self._check_param(params, "models")
        self._check_param(params, "weights")

        params["type"] = "mixture"
        params["family"] = "mixture"
        params["likelihood_f"] = self.prob
        params["loglikelihood_f"] = self.log_prob
        super(CMixtureModel, self).__init__(params)

        models = params["models"]
        weights = params["weights"]
        assert len(models) == len(weights), "len(models)=%d , len(weights)=%d" % (len(models), len(weights))
        self.models = models
        self.weights = weights
        self.set_weights(np.array(weights))

    def set_weights(self, weights):
        assert len(self.models) == len(weights), "len(models)=%d , len(weights)=%d" % (len(self.models), len(weights))

        # Address NaN or negative
        nan_indices = np.logical_or(np.isnan(weights), weights <= 0)
        if np.any(nan_indices):
            print("CMixtureModel. WARNING! There are NaN, negative or zero weights. Those models won't be used",
                  file=sys.stderr)
            weights[nan_indices] = 0

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
            llikelihood[i] = self.models[i].log_prob(data)

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
            # Select the sampling proposal
            idx = np.argmax(np.random.multinomial(1, self.weights))
            q = self.models[idx]

            # Sample from the sampling proposal
            x = q.sample(1)
            res = np.concatenate((res, x)) if res.size else x
        return res

    def condition(self, dist):
        raise NotImplementedError

    def marginal(self, dim):
        raise NotImplementedError


if __name__ == "__main__":
    from ais_benchmarks.distributions import CMultivariateUniform
    from ais_benchmarks.distributions import CMultivariateNormal
    from ais_benchmarks.distributions import CMultivariateExponential
    from matplotlib import pyplot as plt

    # Model 1: Uniform
    center = np.array([10.0])
    radius = np.array([0.5])
    dist1 = CMultivariateUniform({"center": center, "radius": radius})

    # Model 2: Exponential
    dist2 = CMultivariateExponential({"mean": np.array([1.])})

    # Model 3: Normal
    dist3 = CMultivariateNormal({"mean": np.array([3.0]), "sigma": np.array([[0.1]])})

    models = [dist1, dist2, dist3]
    weights = [.2, .6, .2]

    plt.figure()
    plt.title('MixtureModel({%s, %s, %s})' %
              (str(dist1.type), str(dist2.type), str(dist3.type)))
    dist = CMixtureModel({"models": models, "weights": weights, "dims": 1, "support": [-1, 12]}, )
    dist.draw(plt.gca())
    plt.show(block=True)
