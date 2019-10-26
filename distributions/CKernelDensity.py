import numpy as np
from distributions.CMixtureModel import CMixtureModel
from distributions.CMultivariateNormal import CMultivariateNormal
from sklearn.neighbors.kde import KernelDensity
# from scipy.stats import gaussian_kde


# class CKernelDensity:
#     def __init__(self, samples, sample_weight, bw=0.1):
#         # self.kde = gaussian_kde(samples)
#
#         # self.kde = KernelDensity(bandwidth=bw, kernel="epanechnikov")
#         self.kde = KernelDensity(bandwidth=bw, kernel="gaussian")
#         self.kde.fit(samples, sample_weight=sample_weight)
#
#     def log_prob(self, samples):
#         return self.kde.score_samples(samples)
#
#         # return self.kde(samples)

class CKernelDensity:
    def __init__(self, samples, sample_weight, bw=0.1):
        self.samples = samples
        self.weights = sample_weight
        self.bw = bw
        self.model = None
        self.fit()

    def fit(self):
        models = []
        for x in self.samples:
            cov = np.ones(len(self.samples[0])) * self.bw
            if x.shape:
                models.append(CMultivariateNormal(x, np.diag(cov)))
            else:
                models.append(CMultivariateNormal(np.array([x]), np.diag(cov)))
        self.model = CMixtureModel(models, np.exp(self.weights))

    def log_prob(self, data):
        return self.logprob(data)

    def logprob(self, data):
        return np.log(self.prob(data))

    def prob(self, data):
        return self.model.prob(data)
