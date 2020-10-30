import unittest
import numpy as np
import distributions as d
from scipy.stats import multivariate_normal
from tests.distributions.test_distribution_template import CDistributionsTests


class CMultivariateNormalTests(CDistributionsTests):
    __test__ = True
    is_symmetric = True
    dists = list()
    prob_tests = list()

    ############################################
    # Configure 1D Tests. Batched and unbatched.
    ############################################
    mean = np.array([0.0])
    sigma = np.diag([1.5])
    dist1 = d.CMultivariateNormal({"mean": mean, "sigma": sigma})
    dists.append(dist1)

    # Define tested points as triplets: (distribution, x, result)
    # Unbatched test
    x = np.array([0])
    prob_gt = multivariate_normal.pdf(x, mean=mean, cov=sigma)
    prob_tests.append((dist1, x, prob_gt))

    # Batched test
    x = np.array([[0], [0.2], [-2], [-.2]])
    prob_gt_batch = multivariate_normal.pdf(x, mean=mean, cov=sigma).reshape(len(x), 1)
    prob_tests.append((dist1, x, prob_gt_batch))

    ############################################
    # Configure 3D Tests. Batched and unbatched.
    ############################################
    mean = np.array([0.0, 0.0, 0.0])
    sigma = np.diag([1.5, 1.5, 1.5])
    dist2 = d.CMultivariateNormal({"mean": mean, "sigma": sigma})
    dists.append(dist2)

    # Unbatched test
    x = np.array([0, 0, 0])
    prob_gt = multivariate_normal.pdf(x, mean=mean, cov=sigma)
    prob_tests.append((dist2, x, prob_gt))

    # Batched test
    x = np.array([[0, 0, 0], [0, .2, 0], [-2, 0, .3], [-.2, .1, .3]])
    prob_gt = multivariate_normal.pdf(x, mean=mean, cov=sigma).reshape(len(x), 1)
    prob_tests.append((dist2, x, prob_gt))


if __name__ == '__main__':
    unittest.main()

