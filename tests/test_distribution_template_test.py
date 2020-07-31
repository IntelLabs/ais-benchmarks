import unittest
import numpy as np


class CDistributionsTests(unittest.TestCase):
    """
    Directions to write tests for probability distributions

    - Test batched and unbatched
        - Sample method (check dimensions of generated samples are correct)
        - prob and log_prob computations.
        - test that prob is zero outside the support.
        - condition (check that samples and probs are computed correctly after conditioning)

    - Test with multiple dimensions

    - Test other methods
        - marginal
        - integral:
            - Test that the integral over the support is one.
            - Test known values for the integral with multiple dimensions.

    - If testing a kernel test for symmetry
    """
    __test__ = False
    is_symmetric = False
    dists = list()
    prob_tests = list()

    def test_sampling(self):
        for dist in self.dists:
            nsamples = 1000
            samples = dist.sample(nsamples)
            print("%s. %dD. Sample generation shape test." % (dist.__class__.__name__, dist.dims))
            self.assertTrue(len(samples) == nsamples)
            self.assertTrue(samples.shape == (nsamples, dist.dims))

    def test_support_integral(self):
        for dist in self.dists:
            integral = dist.integral(dist.support()[0], dist.support()[1])
            print("%s. %dD. Support integral test. Integral: %f" % (dist.__class__.__name__, dist.dims, integral))
            self.assertTrue(np.allclose(integral, 1.0, atol=0.01))

    def test_probs(self):
        for pt in self.prob_tests:
            prob = pt[0].prob(pt[1])
            prob_gt = pt[2]
            print("%s. %dD. Prob test. prob(%s) = %s. Ground truth: %s" %
                  (pt[0].__class__.__name__, pt[0].dims, str(pt[1].flatten()), str(prob.flatten()), str(pt[2].flatten())))
            self.assertTrue(np.allclose(prob, prob_gt))

            # Test for symmetry
            if self.is_symmetric:
                prob = pt[0].prob(pt[0].loc - pt[1] - pt[0].loc)
                prob_gt = pt[2]
                print("%s. %dD. Prob symmetry test. prob(%s) = %s. Ground truth: %s" %
                      (pt[0].__class__.__name__, pt[0].dims, str(-pt[1].flatten()), str(prob.flatten()),
                       str(pt[2].flatten())))
                self.assertTrue(np.allclose(prob, prob_gt))

    # def test_symmetry_integral(self):
    #     for dist in self.dists:
    #         integral = dist.integral(dist.support()[0], dist.support()[1])
    #         print("%s. %dD. Support integral test. Integral: %f" % (dist.__class__.__name__, dist.dims, integral))
    #         self.assertTrue(np.allclose(integral, 1.0, atol=0.01))
