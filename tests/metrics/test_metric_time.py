import unittest
import numpy as np
import time

from metrics.performance import CElapsedTime

class CMetricsTests(unittest.TestCase):
    """
    Directions to write tests for metrics

    - Test pre and post methods
    - Test compute method with known inputs and results
    - Test reset by resetting and computing again the metric looking for same result
    """
    __test__ = True
    metric = CElapsedTime()

    def test_pre(self):
        self.metric.pre()

    def test_post(self):
        self.metric.post()
        self.metric.reset()

    def test_compute(self):
        delta_t = 0.01
        n_samples = 200
        vals = np.zeros(n_samples)
        for i in range(n_samples):
            self.metric.pre()
            time.sleep(delta_t)
            self.metric.post()
            vals[i] = self.metric.compute()
            self.metric.reset()
        overhead = (vals.mean() - delta_t) / n_samples
        print("\nTesting time measurement metric. Overhead: ", overhead, "Precision:", vals.std())
        self.assertLess(vals.std(), 1e-3, msg="Time measurement not accurate up to millisecond. At least millisecond accuracy is required for proper time metric")
        self.assertLess(overhead, 1e-5, msg="Time measurement overhead is above the recommended value of 10 usecs")
        print("WARNING! Time measurement not accurate up to microsecond") if vals.std() > 1e-6 else None
        print("WARNING! Time measurement not accurate up to nanosecond") if vals.std() > 1e-9 else None

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
        print("Difference of %d measurements that should be the same. Mean: %.9f Std:%.9f" % (n_samples, diffs.mean(), diffs.std()))
        self.assertAlmostEqual(diffs.mean(), 0.0, places=3, msg="Time measurement is not deterministic at least to the millisecond scale")
        print("WARNING! Time measurement not deterministic up to microsecond") if np.fabs(diffs.mean()) > 1e-6 else None
        print("WARNING! Time measurement not deterministic up to nanosecond") if np.fabs(diffs.mean()) > 1e-9 else None
