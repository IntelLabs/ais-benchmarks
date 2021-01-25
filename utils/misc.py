import numpy as np
from threading import Thread
from queue import Queue, Empty

from sampling_methods.base import make_grid
from distributions.mixture.CGaussianMixtureModel import CGaussianMixtureModel

class CNonBlockingStreamReader:
    def __init__(self, stream):
        self._s = stream
        self._q = Queue()

        def _addqueue(s, q):
            while True:
                l = s.readline()
                if l:
                    q.put(l)

        self._t = Thread(target=_addqueue, args=(self._s, self._q))
        self._t.daemon = True
        self._t.start()

    def read_last_and_clear(self, keep_last=True):
        val = None
        while not self._q.empty():
            val = self._q.get()

        if val is not None and keep_last:
            self._q.put(val)
        return val

    def read(self):
        try:
            return self._q.get(block=False)
        except Empty:
            return ""


def generateRandomGMM(space_min, space_max, num_gaussians, sigma_min=(0.04, 0.04, 0.04), sigma_max=(0.05, 0.05, 0.05)):
    means = []
    covs = []

    for _ in range(num_gaussians):
        mu = np.random.uniform(space_min, space_max)
        sigma = np.random.uniform(sigma_min, sigma_max)
        means.append(mu)
        covs.append(sigma)
    gmm = CGaussianMixtureModel(means, covs)
    return gmm


def generateEggBoxGMM(space_min, space_max, delta, sigma):
    [means, dim, shape] = make_grid(space_min, space_max, delta)
    covs = np.array(np.ones_like(means) * sigma)

    gmm = CGaussianMixtureModel(means, covs)
    return gmm


def time_to_hms(total_time):
    hours = total_time // 3600
    mins = (total_time - hours * 3600) // 60
    secs = total_time - hours * 3600 - mins * 60
    return hours, mins, secs