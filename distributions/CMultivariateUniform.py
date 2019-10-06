import numpy as np


class CMultivariateUniform:
    def __init__(self, center, radius):
        self.dims = len(center)

        self.center = center
        self.radius = radius
        self.volume = np.prod(radius*2)

    def sample(self, n_samples=1):
        minval = self.center - self.radius
        maxval = self.center + self.radius
        # TODO: Fix the size problem here and sample in one instruction
        res = np.random.uniform(low=minval, high=maxval, size=None)
        for i in range(1, n_samples):
            res = np.vstack((res, np.random.uniform(low=minval, high=maxval, size=None)))
        return res.reshape(n_samples, self.dims)

    def logprob(self, samples):
        return np.log(self.prob(samples))

    # TODO: MAKE THIS WORK FOR THE MULTIDIMENSIONAL CASE.
    def prob(self, samples):
        if len(samples.shape) == 1:
            samples = samples.reshape(1, self.dims, 1)
        elif len(samples.shape) == 2:
            samples = samples.reshape(len(samples), self.dims, 1)
        else:
            raise ValueError("Shape of samples does not match self.dims")

        min_val = self.center - self.radius
        max_val = self.center + self.radius
        inliers = np.all(np.logical_and(min_val < samples, samples < max_val), axis=1)  # Select the inliers if all the coordinates are in range
        res = np.ones(len(samples)) / self.volume
        res[np.logical_not(inliers.flatten())] = 0
        return res


if __name__ == "__main__":
    import torch
    import time
    from torch import DoubleTensor as t_tensor

    mean = np.array([0, 0, 0])
    cov = np.array([0.1, 1, 1])

    dist1 = torch.distributions.Uniform(t_tensor(mean), t_tensor(cov))
    dist2 = CMultivariateUniform(mean, cov)

    sample = dist2.sample(10000)
    t1 = time.time()
    probs1 = dist1.log_prob(t_tensor(sample))
    probs1_time = time.time() - t1
    # print("Logprob torch:",probs1)
    print("Logprob torch time: ", probs1_time)
    t1 = time.time()
    probs2 = dist2.log_prob(sample)
    probs2_time = time.time() - t1
    # print("Logprob np:",probs2)
    print("Logprob np time: ", probs2_time)
    print("Diff: ", np.sum(np.abs(probs1.numpy()-probs2)))
