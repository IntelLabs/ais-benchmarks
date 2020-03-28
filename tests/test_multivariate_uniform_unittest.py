import unittest
import numpy as np
import distributions as d


class CDistributionsTests(unittest.TestCase):
    """
    Directions to write tests for probability distributions

    - Test batched and unbatched
        - Sample method (just check dimensions are correct)
        - prob and log_prob computations
        - condition (check that samples and probs are computed correctly after conditioning)

    - Test with multiple dimensions

    - Test other methods
        - marginal
        - integral
    """


class CMultivariateUniformTests(unittest.TestCase):
    def test_CMultivariateUniform_sample_1d(self):
        nsamples = 100
        center = np.array([0.0])
        radius = np.array([1.5])
        dist = d.CMultivariateUniform({"center": center, "radius": radius})
        samples = dist.sample(nsamples)
        self.assertTrue(len(samples) == nsamples)
        self.assertTrue(samples.shape == (nsamples, dist.dims))
        print("CMultivariateUniform. 1D. Sample generation shape test.")

    def test_CMultivariateUniform_sample_5d(self):
        nsamples = 100
        center = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        radius = np.array([1.5, 1.5, 1.5, 1.5, 1.5])
        dist = d.CMultivariateUniform({"center": center, "radius": radius})
        samples = dist.sample(nsamples)
        self.assertTrue(len(samples) == nsamples)
        self.assertTrue(samples.shape == (nsamples, dist.dims))
        print("CMultivariateUniform. 5D. Sample generation shape test.")

    def test_CMultivariateUniform_prob_1d(self):
        center = np.array([0.0])
        radius = np.array([1.5])
        dist = d.CMultivariateUniform({"center": center, "radius": radius})
        prob = dist.prob(np.array([0]))
        prob_gt = 1/(np.prod(2*radius))
        print("CMultivariateUniform. 1D. No Batch. prob(0) = %5.3f, should be %5.3f" % (prob, prob_gt))
        self.assertTrue(np.allclose(prob, prob_gt))

    def test_CMultivariateUniform_prob_1d_batch(self):
        center = np.array([0.0])
        radius = np.array([1.5])
        dist = d.CMultivariateUniform({"center": center, "radius": radius})
        x = np.array([[0], [0.2], [-2], [-.2]])
        prob = dist.prob(x)
        prob_gt = 1/(np.prod(2*radius))
        prob_gt_batch = np.array([[prob_gt], [prob_gt], [0], [prob_gt]])
        print("CMultivariateUniform. 1D. Batch. prob(%s) = %s, should be %s" %
              (str(x.flatten()), str(prob.flatten()), str(prob_gt_batch.flatten())))
        self.assertTrue(np.allclose(prob, prob_gt_batch))

    def test_CMultivariateUniform_integral_1d(self):
        center = np.array([0.0])
        radius = np.array([0.5])
        dist = d.CMultivariateUniform({"center": center, "radius": radius})
        print("CMultivariateUniform. Definite integral(0, 1) = %5.3f, should be 0.5" % dist.integral(0, 1))
        self.assertTrue(np.allclose(dist.integral(0, 1), 0.5))
        print("CMultivariateUniform. Definite integral(-1, 1) = %5.3f, should be 1.0" % dist.integral(-1, 1))
        self.assertTrue(np.allclose(dist.integral(-1, 1), 1.0))

        integral = dist.integral(dist.support()[0], dist.support()[1])
        print("CMultivariateUniform. Integral on the support = %5.3f, should be 1.0" % integral)
        self.assertTrue(np.allclose(integral, np.array([1.0])))

    def test_CMultivariateUniform_prob_3d(self):
        # Each dimension has different radius
        center = np.array([0.0, 0.0, 0.0])
        radius = np.array([0.4, 0.5, 0.7])
        dist = d.CMultivariateUniform({"center": center, "radius": radius})
        prob = dist.prob(np.array([0, 0, 0]))
        prob_gt = 1/(np.prod(2*radius))
        print("CMultivariateUniform. 3D. No Batch. Dimension-wise radii. prob(0) = %5.3f, should be %5.3f" % (prob, prob_gt))
        self.assertTrue(np.allclose(prob, prob_gt))

        # All dimensions have same radius
        center = np.array([0.0, 0.0, 0.0])
        radius = np.array([0.4])
        dist = d.CMultivariateUniform({"center": center, "radius": radius})
        prob = dist.prob(np.array([0, 0, 0]))
        prob_gt = 1/((2*radius) ** dist.dims)
        print("CMultivariateUniform. 3D. No Batch. Same radii per dimension. prob(0) = %5.3f, should be %5.3f" % (prob.flatten(), prob_gt.flatten()))
        self.assertTrue(np.allclose(prob, prob_gt))

    def test_CMultivariateUniform_prob_3d_batch(self):
        # Each dimension has different radius
        center = np.array([0.0, 0.0, 0.0])
        radius = np.array([1.0, 1.5, 5.5])
        dist = d.CMultivariateUniform({"center": center, "radius": radius})
        x = np.array([[0, 0, 0], [0, .2, 0], [-2, 0, .3], [-.2, .1, .3]])
        prob = dist.prob(x)
        prob_gt = 1/(np.prod(2*radius))
        prob_gt_batch = np.array([[prob_gt], [prob_gt], [0], [prob_gt]])
        print("CMultivariateUniform. 3D. Batch. Dimension-wise radii. prob(%s) = %s, should be %s" %
              (str(x.flatten()), str(prob.flatten()), str(prob_gt_batch.flatten())))
        self.assertTrue(np.allclose(prob, prob_gt_batch))

        # All dimensions have same radii
        center = np.array([0.0, 0.0, 0.0])
        radius = np.array([1.5])
        dist = d.CMultivariateUniform({"center": center, "radius": radius})
        x = np.array([[0, 0, 0], [0, .2, 0], [-2, 0, .3], [-.2, .1, .3]])
        prob = dist.prob(x)
        prob_gt = 1/((2 * radius) ** dist.dims)
        prob_gt_batch = np.array([prob_gt, prob_gt, [0], prob_gt])
        print("CMultivariateUniform. 3D. Batch. Same radii per dimension. prob(%s) = %s, should be %s" %
              (str(x.flatten()), str(prob.flatten()), str(prob_gt_batch.flatten())))
        self.assertTrue(np.allclose(prob, prob_gt_batch))

    def test_CMultivariateUniform_integral_3d(self):
        center = np.array([0.0, 0.0, 0.0])
        radius = np.array([0.5, 0.5, 0.5])
        dist = d.CMultivariateUniform({"center": center, "radius": radius})
        print("CMultivariateUniform. Definite integral(0, 1) = %5.3f, should be 0.125" % dist.integral(0, 1))
        self.assertTrue(np.allclose(dist.integral(0, 1), 0.125))
        print("CMultivariateUniform. Definite integral(-1, 1) = %5.3f, should be 1.0" % dist.integral(-1, 1))
        self.assertTrue(np.allclose(dist.integral(-1, 1), 1.0))

        integral = dist.integral(dist.support()[0], dist.support()[1])
        print("CMultivariateUniform. Integral on the support = %5.3f, should be 1.0" % integral)
        self.assertTrue(np.allclose(integral, np.array([1.0])))


if __name__ == '__main__':
    unittest.main()
