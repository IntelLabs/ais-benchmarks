import unittest
import numpy as np
import ais_benchmarks.distributions as d
from tests.distributions.test_distribution_template import CDistributionsTests


class CMultivariateUniformTests(CDistributionsTests):
    __test__ = True
    is_symmetric = True
    prob_tests = list()
    dists = list()

    ############################################
    # Configure 1D Tests. Batched and unbatched.
    ############################################
    center = np.array([0.1])
    radius = np.array([1.5])
    dist1 = d.CMultivariateUniform({"center": center, "radius": radius})
    dists.append(dist1)

    # Define tested points as triplets: (distribution, x, result)
    # Unbatched test
    x = np.array([0])
    prob_gt = 1 / (np.prod(2 * radius))
    prob_tests.append((dist1, x, prob_gt))

    # Batched test
    x = np.array([[0], [0.2], [-2], [-.2]])
    prob_gt = 1 / (np.prod(2 * radius))
    prob_gt_batch = np.array([prob_gt, prob_gt, 0, prob_gt])
    prob_tests.append((dist1, x, prob_gt_batch))

    ############################################
    # Configure 3D Tests. Batched and unbatched.
    ############################################
    center = np.array([0.1, -0.2, 0.0])
    radius = np.array([0.4, 0.5, 0.7])
    dist2 = d.CMultivariateUniform({"center": center, "radius": radius})
    dists.append(dist2)

    # Unbatched test
    x = np.array([0, 0, 0])
    prob_gt = 1 / (np.prod(2 * radius))
    prob_tests.append((dist2, x, prob_gt))

    # Batched test
    x = np.array([[0, 0, 0], [0, .2, 0], [-2, 0, .3], [-.2, .1, .3]])
    prob_gt = 1 / (np.prod(2 * radius))
    prob_gt_batch = np.array([prob_gt, prob_gt, 0, prob_gt])
    prob_tests.append((dist2, x, prob_gt_batch))

    # Batched test: All dimensions have same radius
    center = np.array([0.0, 0.0, 0.0])
    radius = np.array([1.5])
    dist3 = d.CMultivariateUniform({"center": center, "radius": radius})
    x = np.array([[0, 0, 0], [0, .2, 0], [-2, 0, .3], [-.2, .1, .3]])
    prob = dist3.prob(x)
    prob_gt = 1 / ((2 * float(radius)) ** dist3.dims)
    prob_gt_batch = np.array([prob_gt, prob_gt, 0, prob_gt])
    prob_tests.append((dist3, x, prob_gt_batch))


if __name__ == '__main__':
    unittest.main()
