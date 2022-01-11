import unittest
import numpy as np
import time

from ais_benchmarks.metrics.performance import CElapsedTime


class CMetricsTests(unittest.TestCase):
    """
    Directions to write tests for metrics

    - Test pre and post methods
    - Test compute method with known inputs and results
    - Test reset by resetting and computing again the metric should yield
      the same result
    """
    __test__ = True
    metric = CElapsedTime()

    def test_pre(self):
        self.metric.pre()

    def test_post(self):
        self.metric.post()
        self.metric.reset()

    def test_compute(self):
        self.metric = CElapsedTime()
        delta_t = 0.01
        n_samples = 200
        vals = np.zeros(n_samples)
        for i in range(n_samples):
            self.metric.pre()
            time.sleep(delta_t)
            self.metric.post()
            vals[i] = self.metric.compute()
            self.metric.reset()
        overhead = (vals.mean() - delta_t)
        print("\nTesting time measurement metric. Overhead: ", overhead, "Precision:", vals.std())
        self.assertLess(vals.std(), 1e-3, msg="Time measurement not accurate up to millisecond. At least millisecond accuracy is required for proper time metric")
        self.assertLess(overhead, 1e-5, msg="Time measurement overhead is above the recommended value of 10 usecs")

        if vals.std() > 1e-6:
            print("WARNING! Time measurement not accurate up to microsecond")
        if vals.std() > 1e-9:
            print("WARNING! Time measurement not accurate up to nanosecond")

    def test_reset(self):
        delta_t = 0.01
        n_samples = 100
        res = np.zeros(n_samples)
        for i in range(n_samples):
            self.metric.pre()
            time.sleep(delta_t)
            self.metric.post()
            res[i] = self.metric.compute()
            self.metric.reset()
        diffs = res - res.mean()
        print(f"Difference of {n_samples} measurements that should be the same."
              f" Mean: {diffs.mean():.9f} Std:{diffs.std():.9f}")
        self.assertAlmostEqual(diffs.mean(), 0.0, places=6,
                               msg="Time measurement is not deterministic at "
                                   "least to the microsecond scale")

        if np.fabs(diffs.mean()) > 1e-6:
            print("WARNING! Time metric not deterministic up to microsecond.")

        if np.fabs(diffs.mean()) > 1e-9:
            print("WARNING! Time metric not deterministic up to nanosecond.")
