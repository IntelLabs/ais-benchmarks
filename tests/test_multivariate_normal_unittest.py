import unittest
import numpy as np
import distributions as d
from scipy.stats import multivariate_normal


class CMultivariateNormalTests(unittest.TestCase):
    def test_CMultivariateNormal_sample_1d(self):
        nsamples = 100
        mean = np.array([0.0])
        sigma = np.diag([1.5])
        dist = d.CMultivariateNormal({"mean": mean, "sigma": sigma})
        samples = dist.sample(nsamples)
        self.assertTrue(len(samples) == nsamples)
        self.assertTrue(samples.shape == (nsamples, dist.dims))
        print("CMultivariateNormal. 1D. Sample generation shape test.")

    def test_CMultivariateNormal_sample_5d(self):
        nsamples = 100
        mean = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        sigma = np.diag([1.5, 1.5, 1.5, 1.5, 1.5])
        dist = d.CMultivariateNormal({"mean": mean, "sigma": sigma})
        samples = dist.sample(nsamples)
        self.assertTrue(len(samples) == nsamples)
        self.assertTrue(samples.shape == (nsamples, dist.dims))
        print("CMultivariateNormal. 5D. Sample generation shape test.")

    def test_CMultivariateNormal_prob_1d(self):
        mean = np.array([0.0])
        sigma = np.diag([1.0])
        dist = d.CMultivariateNormal({"mean": mean, "sigma": sigma})
        x = np.array([0])
        prob = dist.prob(x)
        prob_gt = multivariate_normal.pdf(x, mean=mean, cov=sigma)
        print("CMultivariateNormal. 1D. No Batch. prob(%s) = %5.3f, should be %5.3f" % (str(x), prob, prob_gt))
        self.assertTrue(np.allclose(prob, prob_gt))

    def test_CMultivariateNormal_prob_1d_batch(self):
        mean = np.array([0.0])
        sigma = np.diag([1.5])
        dist = d.CMultivariateNormal({"mean": mean, "sigma": sigma})
        x = np.array([[0], [0.2], [-2], [-.2]])
        prob = dist.prob(x)
        prob_gt_batch = multivariate_normal.pdf(x, mean=mean, cov=sigma).reshape(prob.shape)
        print("CMultivariateNormal. 1D. Batch. prob(%s) = %s, should be %s" %
              (str(x.flatten()), str(prob.flatten()), str(prob_gt_batch.flatten())))
        self.assertTrue(np.allclose(prob, prob_gt_batch))

    def test_CMultivariateNormal_integral_1d(self):
        mean = np.array([0.0])
        sigma = np.diag([0.5])
        dist = d.CMultivariateNormal({"mean": mean, "sigma": sigma})

        start = np.array([-1.96 * np.sqrt(sigma)])
        end = np.array([1.96 * np.sqrt(sigma)])
        integral_gt = np.array([.95])
        integral = dist.integral(start, end)
        print("CMultivariateNormal. Definite integral(%s, %s) = %5.3f, should be %5.3f" % (str(start), str(end), integral, integral_gt))
        self.assertTrue(np.allclose(integral, integral_gt, atol=.01))  # Be forgiving with the tolerance

    def test_CMultivariateNormal_prob_3d(self):
        mean = np.array([0.3, -1.0, 0.0])
        sigma = np.diag([0.4, 0.5, 0.7])
        dist = d.CMultivariateNormal({"mean": mean, "sigma": sigma})
        x = np.array([0, 0, 0])
        prob = dist.prob(x)
        prob_gt = multivariate_normal.pdf(x, mean=mean, cov=sigma)
        print("CMultivariateNormal. 3D. No Batch. prob(%s) = %5.3f, should be %5.3f" % (str(x), prob, prob_gt))
        self.assertTrue(np.allclose(prob, prob_gt))

    def test_CMultivariateNormal_prob_3d_batch(self):
        mean = np.array([0.3, -1.0, 0.0])
        sigma = np.diag([0.4, 0.5, 0.7])
        dist = d.CMultivariateNormal({"mean": mean, "sigma": sigma})
        x = np.array([[0, 0, 0], [0, .2, 0], [-2, 0, .3], [-.2, .1, .3]])
        prob = dist.prob(x)
        prob_gt = multivariate_normal.pdf(x, mean=mean, cov=sigma).reshape(prob.shape)
        print("CMultivariateNormal. 3D. Batch. prob(%s) = %s, should be %s" %
              (str(x.flatten()), str(prob.flatten()), str(prob_gt.flatten())))
        self.assertTrue(np.allclose(prob, prob_gt))

    def test_CMultivariateNormal_integral_3d(self):
        mean = np.array([0.0, 0.0, 0.0])
        sigma = np.diag([0.5, 0.5, 0.5])
        dist = d.CMultivariateNormal({"mean": mean, "sigma": sigma})

        start = np.array([0.0, 0.0, 0.0])
        end = np.array([6.0, 6.0, 6.0])
        integral_gt = np.array([1.0 / 8.0])
        integral = dist.integral(start, end)
        print("CMultivariateNormal. Definite integral(%s, %s) = %5.3f, should be %5.3f" % (str(start), str(end), integral, integral_gt))
        self.assertTrue(np.allclose(integral, integral_gt, atol=.01))  # Be forgiving with the tolerance


if __name__ == '__main__':
    unittest.main()

