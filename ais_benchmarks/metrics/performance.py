"""
Performance metrics will evaluate computation complexity, memory consumption and efficiency (power consumption)
"""

import time
import gc
import numpy as np
from guppy import hpy

from metrics.base import CMetric


class CElapsedTime(CMetric):
    def __init__(self):
        super().__init__()
        self.name = "time"
        self.type = "performance"
        self.t_ini = 0
        self.t_end = 0
        self.overhead = 0
        print("CElapsedTime initialize metric overhead estimate...")

        # Compute measurement overhead to subtract it when measuring.
        n_samples = 200
        delta_t = 0.01
        vals = np.zeros(n_samples)
        for i in range(n_samples):
            self.pre()
            time.sleep(delta_t)
            self.post()
            vals[i] = self.compute()
            self.reset()
        self.overhead = (vals.mean() - delta_t) / n_samples
        print("CElapsedTime overhead estimate: ", self.overhead)

    def pre(self, **kwargs):
        self.t_ini = time.time()

    def post(self, **kwargs):
        self.t_end = time.time()
        self.value += self.t_end - self.t_ini - self.overhead

    def compute(self, **kwargs):
        return self.value

    def reset(self, **kwargs):
        self.t_ini = 0
        self.t_end = 0
        self.value = 0


class CMemoryUsage(CMetric):
    def __init__(self):
        super().__init__()
        self.name = "memory"
        self.type = "performance"
        self.mem_ini = 0
        self.mem_end = 0
        self.mem_cum = 0
        self.hpy = hpy()

    def pre(self, **kwargs):
        gc.collect()
        self.mem_ini = self.hpy.heap().size

    def post(self, **kwargs):
        gc.collect()
        self.mem_end = self.hpy.heap().size
        self.mem_cum += self.mem_end - self.mem_ini
        self.value = self.mem_cum / 10**6

    def compute(self, **kwargs):
        return self.value

    def reset(self, **kwargs):
        self.mem_ini = 0
        self.mem_end = 0
        self.mem_cum = 0
        gc.collect()

