import unittest
import numpy as np

from metrics.divergences import CJSDivergence
from distributions.parametric.CMultivariateNormal import CMultivariateNormal
from distributions.parametric.CMultivariateUniform import CMultivariateUniform


class CMetricsTests(unittest.TestCase):
    """
    Directions to write tests for metrics

    - Test pre and post methods
    - Test compute method with known inputs and results
    - Test reset by resetting and computing again the metric looking for same result
    """
    __test__ = True
    metric = CJSDivergence()

    def test_pre(self):
        print("Test pre method")
        self.metric.pre()

    def test_post(self):
        print("Test post and reset method")
        self.metric.post()
        self.metric.reset()

    def do_test(self, p, q, expected_res, tolerance=0.01):
        res = self.metric.compute(p=p, q=q, nsamples=100000)
        print("Test JSD divergence. JSD:%.6f. Expected: %.6f. Error: %.3f%%" %
              (res, expected_res, np.fabs(100*(res-expected_res)/expected_res)))

        self.assertAlmostEqual(res, expected_res, delta=expected_res*tolerance,
                               msg="Divergence metric should be within %.1f%% of the expected value %f" %
                                   (tolerance*100, expected_res))

    @staticmethod
    def mvn_kld(mu0, mu1, sigma0, sigma1):
        """
        This is the closed form KLDivergence for two multivariate normal distributions
        """
        return 0.5 * (np.trace(np.linalg.inv(sigma1) @ sigma0) +
                      np.transpose(mu1 - mu0) @ np.linalg.inv(sigma1) @ np.transpose(mu1 - mu0) -
                      len(mu0) + np.log(np.linalg.det(sigma1) / np.linalg.det(sigma0)))

    @staticmethod
    def mvn_jsd(mu0, mu1, sigma0, sigma1):
        """
        No Closed form for the jsd exists. The main problem comes from the distance being computed to the
        midpoint distribution. More info in this dicussion:
        https://stats.stackexchange.com/questions/8634/jensen-shannon-divergence-for-bivariate-normal-distributions
        """
        mu_m = .5 * mu0 + .5 * mu1
        sigmam = np.sqrt(.5 * .5 * sigma0 + .5 * .5 * sigma1)
        """
        This is the closed form JSDivergence for two multivariate normal distributions based on the closed form KL
        """
        from scipy.spatial import distance
        p = CMultivariateNormal({"mean": mu0, "sigma": sigma0})
        q = CMultivariateNormal({"mean": mu1, "sigma": sigma1})
        samples = np.random.uniform(p.support()[0], p.support()[1], size=(1000000, p.dims))
        res = distance.jensenshannon(p.prob(samples), q.prob(samples)) ** 2

        return res

    def test_equal(self):
        # Test distributions are the same
        p = CMultivariateNormal({"mean": np.array([0]), "sigma": np.diag([1])})
        q = CMultivariateNormal({"mean": np.array([0]), "sigma": np.diag([1])})
        self.do_test(p, q, 0)

    def test_similar(self):
        mu0 = np.array([0])
        mu1 = np.array([0.01])
        sigma0 = np.diag([1])
        sigma1 = np.diag([1])
        p = CMultivariateNormal({"mean": mu0, "sigma": sigma0})
        q = CMultivariateNormal({"mean": mu1, "sigma": sigma1})
        expected_res = self.mvn_jsd(mu0, mu1, sigma0, sigma1)
        self.do_test(p, q, expected_res)

    def test_similar2(self):
        mu0 = np.array([0])
        mu1 = np.array([2])
        sigma0 = np.diag([1])
        sigma1 = np.diag([1])
        p = CMultivariateNormal({"mean": mu0, "sigma": sigma0})
        q = CMultivariateNormal({"mean": mu1, "sigma": sigma1})
        expected_res = self.mvn_jsd(mu0, mu1, sigma0, sigma1)
        self.do_test(p, q, expected_res)

    def test_different(self):
        mu0 = np.array([0])
        mu1 = np.array([5])
        sigma0 = np.diag([1])
        sigma1 = np.diag([.1])
        p = CMultivariateNormal({"mean": mu0, "sigma": sigma0})
        q = CMultivariateNormal({"mean": mu1, "sigma": sigma1})
        expected_res = self.mvn_jsd(mu0, mu1, sigma0, sigma1)
        self.do_test(p, q, expected_res)

    def test_different2(self):
        mu0 = np.array([0])
        mu1 = np.array([2])
        sigma0 = np.diag([2])
        sigma1 = np.diag([2])
        p = CMultivariateNormal({"mean": mu0, "sigma": sigma0})
        q = CMultivariateNormal({"mean": mu1, "sigma": sigma1})
        expected_res = self.mvn_jsd(mu0, mu1, sigma0, sigma1)
        self.do_test(p, q, expected_res)

    def test_very_different2(self):
        mu0 = np.array([0])
        mu1 = np.array([10])
        sigma0 = np.diag([.2])
        sigma1 = np.diag([.2])
        p = CMultivariateNormal({"mean": mu0, "sigma": sigma0})
        q = CMultivariateNormal({"mean": mu1, "sigma": sigma1})
        expected_res = self.mvn_jsd(mu0, mu1, sigma0, sigma1)
        self.do_test(p, q, expected_res)

    def test_different_multidim(self):
        mu0 = np.array([0, 0, 0])
        mu1 = np.array([5, 5, 5])
        sigma0 = np.diag([1, 1, 1])
        sigma1 = np.diag([.5, .5, .5])
        p = CMultivariateNormal({"mean": mu0, "sigma": sigma0})
        q = CMultivariateNormal({"mean": mu1, "sigma": sigma1})
        expected_res = self.mvn_jsd(mu0, mu1, sigma0, sigma1)
        self.do_test(p, q, expected_res)

    def test_disjoint(self):
        p = CMultivariateUniform({"center": np.array([0]), "radius": np.array([.1])})
        q = CMultivariateUniform({"center": np.array([1]), "radius": np.diag([.1])})
        self.do_test(p, q, np.inf)

    def test_symmetric(self):
        for _ in range(10):
            ndims = 5
            mu0 = np.array(np.random.random(ndims))
            mu1 = np.array(np.random.random(ndims))
            sigma0 = np.diag(np.full(ndims, .5))
            sigma1 = np.diag(np.full(ndims, .5))
            p = CMultivariateNormal({"mean": mu0, "sigma": sigma0})
            q = CMultivariateNormal({"mean": mu1, "sigma": sigma1})

            res1 = self.metric.compute(p=p, q=q, nsamples=1000000)
            res2 = self.metric.compute(p=q, q=p, nsamples=1000000)
            print("Test JSD divergence. JSD(p||q):%.6f. JSD(q||p):%.6f: . Error: %f" %
                  (res1, res2, np.fabs(res1-res2)))
            self.assertAlmostEqual(res1, res2, delta=np.log(2)*0.01, msg="JSD failed symmetry test.")

    def test_behavior(self):
        ntests = 10
        ndims = 3

        mu0 = np.array(np.random.random(ndims))
        mu1 = np.copy(mu0)
        sigma0 = np.diag(np.full(ndims, .5))
        sigma1 = np.diag(np.full(ndims, .5))

        res = np.zeros(ntests)
        for i in range(ntests):
            p = CMultivariateNormal({"mean": mu0, "sigma": sigma0})
            q = CMultivariateNormal({"mean": mu1, "sigma": sigma1})

            res[i] = self.metric.compute(p=p, q=q, nsamples=1000000)

            mu1 += np.full(ndims, .1)

        print("Test JSD divergence with increasingly disctint means.", np.array_str(res, precision=6))
        for i in range(ntests-1):
            self.assertTrue(res[i]<res[i+1], msg="JSD failed the increasingly disctinct test. Each tested distribution is more different than the previous.")

    def test_reset(self):
        self.metric.reset()
