import unittest
import numpy as np

from metrics.divergences import CKLDivergence
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
    metric = CKLDivergence()

    def test_pre(self):
        print("Test pre method")
        self.metric.pre()

    def test_post(self):
        print("Test post and reset method")
        self.metric.post()
        self.metric.reset()

    def do_test(self, p, q, expected_res):
        res = self.metric.compute(p=p, q=q, nsamples=100000)
        print("Test KL divergence. KLD:%.6f. Expected: %.6f" % (res, expected_res))
        self.assertAlmostEqual(res, expected_res, places=3, msg="Divergence metric should be 0 for distributions that are the same")

    @staticmethod
    def mvn_kld(mu0, mu1, sigma0, sigma1):
        """
        This is the closed form KLDivergence for two multivariate normal distributions
        """
        return 0.5 * (np.trace(np.linalg.inv(sigma1) @ sigma0) +
                      np.transpose(mu1 - mu0) @ np.linalg.inv(sigma1) @ np.transpose(mu1 - mu0) -
                      len(mu0) + np.log(np.linalg.det(sigma1) / np.linalg.det(sigma0)))


    def test_equal(self):
        # Test distributions are the same
        p = CMultivariateNormal({"mean": np.array([0]), "sigma":np.diag([1])})
        q = CMultivariateNormal({"mean": np.array([0]), "sigma":np.diag([1])})
        self.do_test(p, q, 0)

    def test_similar(self):
        mu0 = np.array([0])
        mu1 = np.array([0.01])
        sigma0 = np.diag([1])
        sigma1 = np.diag([1])
        p = CMultivariateNormal({"mean": mu0, "sigma":sigma0})
        q = CMultivariateNormal({"mean": mu1, "sigma":sigma1})
        expected_res = self.mvn_kld(mu0, mu1, sigma0, sigma1)
        self.do_test(p, q ,expected_res)

    def test_similar2(self):
        mu0 = np.array([0])
        mu1 = np.array([0.1])
        sigma0 = np.diag([1])
        sigma1 = np.diag([1])
        p = CMultivariateNormal({"mean": mu0, "sigma":sigma0})
        q = CMultivariateNormal({"mean": mu1, "sigma":sigma1})
        expected_res = self.mvn_kld(mu0, mu1, sigma0, sigma1)
        self.do_test(p, q ,expected_res)

    def test_similar3(self):
        mu0 = np.array([0])
        mu1 = np.array([2])
        sigma0 = np.diag([1])
        sigma1 = np.diag([1])
        p = CMultivariateNormal({"mean": mu0, "sigma":sigma0})
        q = CMultivariateNormal({"mean": mu1, "sigma":sigma1})
        expected_res = self.mvn_kld(mu0, mu1, sigma0, sigma1)
        self.do_test(p, q ,expected_res)

    def test_different(self):
        mu0 = np.array([0])
        mu1 = np.array([5])
        sigma0 = np.diag([1])
        sigma1 = np.diag([.1])
        p = CMultivariateNormal({"mean": mu0, "sigma":sigma0})
        q = CMultivariateNormal({"mean": mu1, "sigma":sigma1})
        expected_res = self.mvn_kld(mu0, mu1, sigma0, sigma1)
        self.do_test(p, q, expected_res)

    def test_different_multidim(self):
        mu0 = np.array([0, 0, 0])
        mu1 = np.array([5, 5, 5])
        sigma0 = np.diag([1, 1, 1])
        sigma1 = np.diag([.5, .5, .5])
        p = CMultivariateNormal({"mean": mu0, "sigma":sigma0})
        q = CMultivariateNormal({"mean": mu1, "sigma":sigma1})
        expected_res = self.mvn_kld(mu0, mu1, sigma0, sigma1)
        self.do_test(p, q, expected_res)

    def test_disjoint(self):
        p = CMultivariateUniform({"center": np.array([0]), "radius":np.array([.1])})
        q = CMultivariateUniform({"center": np.array([1]), "radius":np.diag([.1])})
        self.do_test(p, q, np.inf)

    def test_symmetric(self):
        pass


    def test_reset(self):
        self.metric.reset()
