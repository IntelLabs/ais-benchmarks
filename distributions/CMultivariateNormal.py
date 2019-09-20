import numpy as np


class CMultivariateNormal:
    def __init__(self, mean, cov):
        self.dims = len(mean)
        assert cov.shape == (self.dims, self.dims)

        self.det = np.linalg.det(cov)
        assert self.det > 0

        self.mean = mean.reshape(1,self.dims,1)
        self.cov = cov
        self.inv_cov = np.linalg.inv(cov).reshape((1, self.dims, self.dims))
        self.log_det = np.log(self.det)

        self.term1 = - 0.5 * self.dims * np.log(np.pi * 2)
        self.term2 = - 0.5 * self.log_det

    def set_moments(self, mean, cov):
        assert cov.shape == (self.dims, self.dims)

        self.det = np.linalg.det(cov)
        assert self.det > 0

        self.mean = mean.reshape(1,self.dims,1)
        self.cov = cov
        self.inv_cov = np.linalg.inv(cov).reshape((1, self.dims, self.dims))
        self.log_det = np.log(self.det)

    def sample(self, n_samples=1):
        return np.random.multivariate_normal(self.mean.flatten(), self.cov, size=n_samples)

    def logprob(self, samples):
        if len(samples.shape) == 1:
            samples = samples.reshape(1,-1,1)
        elif len(samples.shape) == 2:
            samples = samples.reshape(len(samples), -1, 1)

        diff = self.mean - samples
        term3_1 = np.matmul(np.transpose(diff, axes=(0, 2, 1)), self.inv_cov)
        term3_2 = np.matmul(term3_1, diff)
        term3 = -0.5 * term3_2.flatten()
        return self.term1 + self.term2 + term3

    def prob(self, samples):
        return np.exp(self.logprob(samples))


if __name__ == "__main__":
    import torch
    import time
    from torch import DoubleTensor as t_tensor

    mean = np.array([0, 0, 0])
    cov = np.diag([0.1, 1, 1])

    dist1 = torch.distributions.MultivariateNormal(t_tensor(mean), t_tensor(cov))
    dist2 = CMultivariateNormal(mean, cov)

    sample = dist2.sample(10000)
    t1 = time.time()
    probs1 = dist1.log_prob(sample)
    probs1_time = time.time() - t1
    # print("Logprob torch:",probs1)
    print("Logprob torch time: ", probs1_time)
    t1 = time.time()
    probs2 = dist2.log_prob(sample)
    probs2_time = time.time() - t1
    # print("Logprob np:",probs2)
    print("Logprob np time: ", probs2_time)
    print("Diff: ", np.sum(np.abs(probs1.numpy()-probs2)))
