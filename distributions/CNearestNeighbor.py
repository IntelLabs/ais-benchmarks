from numpy import array as t_tensor
from scipy import spatial

class CNearestNeighbor:
    def __init__(self, samples, sample_weight):
        self.samples = samples
        self.values = sample_weight
        self.kdtree = spatial.KDTree(samples)

    def log_prob(self, samples):
        dist, idx = self.kdtree.query(samples)
        return self.values[idx]
