from sklearn.neighbors.kde import KernelDensity
# from scipy.stats import gaussian_kde


class CKernelDensity:
    def __init__(self, samples, sample_weight, bw=0.1):
        # self.kde = gaussian_kde(samples)

        # self.kde = KernelDensity(bandwidth=bw, kernel="epanechnikov")
        self.kde = KernelDensity(bandwidth=bw, kernel="gaussian")
        self.kde.fit(samples, sample_weight=sample_weight)

    def log_prob(self, samples):
        return self.kde.score_samples(samples)

        # return self.kde(samples)