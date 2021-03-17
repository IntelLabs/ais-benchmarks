import unittest
import numpy as np

from ais_benchmarks.metrics.performance import CMemoryUsage


class CMetricsTests(unittest.TestCase):
    """
    Directions to write tests for metrics

    - Test pre and post methods
    - Test compute method with known inputs and results
    - Test reset by resetting and computing again the metric looking for same result
    """
    __test__ = True
    metric = CMemoryUsage()

    def test_pre(self):
        print("Test pre method")
        self.metric.pre()

    def test_post(self):
        print("Test post and reset method")
        self.metric.post()
        self.metric.reset()

    def test_compute(self):
        num_mbs = [1, 10, 50, 100, 500]
        n_samples = 30
        for num_mb in num_mbs:
            vals = np.zeros(n_samples)
            for i in range(n_samples):
                self.metric.pre()
                dummy_var = np.ones(num_mb * 10**6, dtype=np.byte)
                self.metric.post()
                del dummy_var
                vals[i] = self.metric.compute()
                self.metric.reset()
            print("\nTesting memory measurement metric. Result: ", vals.mean(), "Std:", vals.std(), "should be:", num_mb)
            self.assertAlmostEqual(vals.mean(), num_mb, delta=1./1024.,
                                   msg="Memory measurement not accurate up to KB. At least KB accuracy is required for proper memory metric")

    def test_reset(self):
        num_mb = 10
        n_samples = 100
        vals = np.zeros(n_samples)
        for i in range(n_samples):
            self.metric.pre()
            dummy_var = np.ones(num_mb * 10**6, dtype=np.byte)
            self.metric.post()
            del dummy_var
            vals[i] = self.metric.compute()
            self.metric.reset()
        diffs = vals - vals.mean()
        print("Difference of %d measurements that should be the same. Mean: %.9f Std:%.9f" % (n_samples, diffs.mean(), diffs.std()))
        self.assertAlmostEqual(diffs.mean(), 0.0, places=3, msg="Time measurement is not deterministic at least to the millisecond scale")
        print("WARNING! Memory measurement not deterministic up to Kilobyte") if np.fabs(diffs.mean()) > 1e-3 else None
        print("WARNING! Memory measurement not deterministic up to Byte") if np.fabs(diffs.mean()) > 1e-6 else None
