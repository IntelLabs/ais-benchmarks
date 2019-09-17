import time
import numpy as np
from numpy import array as t_tensor
import itertools

from sampling_methods.base import CSamplingMethod
from distributions.CMultivariateNormal import CMultivariateNormal
from distributions.CMultivariateUniform import CMultivariateUniform


class CTreePyramidNode:
    def __init__(self, center, radius, node_idx, leaf_idx, level, kernel="haar"):
        """
        A node of a tree pyramid contains information about its location
        :param center: location of the center of the node (In the paper: n_c)
        :param radius: radius of the node (In the paper: n_r)
        :param leaf_idx: Index of the node in the tree leaf list. If the node is not a leaf this index is -1
        :param node_idx: Index of the node in the tree node list
        :param level: Distance of the node to the root. Must be consistent with the children and parents.
        """
        self.center = center                        # n_c
        self.radius = radius                        # n_r
        self.children = [None] * (2**len(center))   # n_s
        self.weight = t_tensor(0)                   # n_w
        self.coords = t_tensor([0] * len(center))   # n_x
        self.leaf_idx = leaf_idx
        self.node_idx = node_idx
        self.level = level
        self.value = self.weight * self.radius ** len(center)  # Importance volume used to sort nodes
        self.coords_hist = []
        self.weight_hist = []

        """
        Sampling kernel distribution. This shapes the proposal distribution represented by the tree pyramid.
        """
        self.sampler = CMultivariateUniform(self.center, self.radius)
        if kernel == "normal":
            self.sampler = CMultivariateNormal(self.center, np.diag(t_tensor([self.radius] * len(self.center))))
        elif kernel == "haar":
            self.sampler = CMultivariateUniform(self.center, self.radius)
        else:
            raise ValueError("Unknown kernel type. Must be 'normal' or 'haar'")

    def sample(self):
        """
        This is the Monte Carlo portion of the Quasi-Monte Carlo approach. The sample location (n_x) is obtained
        by a proposal distribution parameterized by the node center (n_c) and radius (n_r).
        """
        self.coords = self.sampler.sample()[0]  # Samplers return a batch of samples. This removes the batch dimension.
        self.coords_hist.append(self.coords)

    def weigh(self, target_d, importance_d, target_f=lambda x: 1):
        """
        This is the importance sampling part where the weight of the sample is updated.

        :param target_d: Target distribution to compute the importance weight. In the paper: \pi(x)
        :param importance_d: Importance distribution used to generate the sample. In the paper: q(x)
        :param target_f: Target f(x) used for importance sampling.
        """
        self.weight = (target_f(self.coords) * target_d.prob(self.coords)) / importance_d.prob(self.coords)
        self.value = self.weight * (self.radius ** len(self.center))
        self.weight_hist.append(self.weight)

    def __lt__(self, other):
        return self.value < other.value

    def is_leaf(self):
        return self.leaf_idx != -1

    def __float__(self):
        return self.value

    def __repr__(self):
        return "[ l_idx:" + str(self.leaf_idx) + " , nidx:" + str(self.node_idx) + " , c:" + str(self.center) + " , r:" + str(self.radius) + " , w:" + str(self.weight) + " ]"


class CTreePyramid:
    def __init__(self, dmin, dmax):
        self.root = CTreePyramidNode(center=(dmin + dmax) / 2, radius=np.max((dmax - dmin) / 2),
                                     leaf_idx=0, node_idx=0, level=0)
        self.nodes = [self.root]
        self.leaves = [self.root]
        self.ndims = len(dmin)

    def expand(self, node):
        assert node.is_leaf(), "Non leaf node expansion is not allowed: " + repr(node)

        ################################################
        # Compute new node centers and radii
        ################################################
        center_coeffs = itertools.product([-1, 1], repeat=len(node.center))

        # TODO: Vectorize this operation
        p_centers = []
        for coeff in center_coeffs:
            offset = (node.radius / 2) * t_tensor(coeff)
            p_centers.append(node.center + offset)

        new_radius = node.radius / 2
        ################################################
        ################################################

        ################################################
        # Create the new child nodes and update the tree
        ################################################
        # Create the new child nodes
        new_nodes = []
        for i, center in enumerate(p_centers):
            new_nodes.append(CTreePyramidNode(center=center, radius=new_radius,
                                              leaf_idx=len(self.leaves) + i - 1, node_idx=len(self.nodes) + i,
                                              level=node.level + 1))

        # The first new node replaces the expanding leaf node in the leaves list
        new_nodes[0].leaf_idx = node.leaf_idx
        self.leaves[node.leaf_idx] = new_nodes[0]
        self.nodes.append(new_nodes[0])
        node.children[0] = new_nodes[0]
        node.leaf_idx = -1

        # The rest get inserted into the tree
        for i, new_n in enumerate(new_nodes[1:]):
            node.children[i+1] = new_n
            new_n.leaf_idx = len(self.leaves)
            self.leaves.append(new_n)
            self.nodes.append(new_n)
        ################################################
        ################################################

        # Return the newly created nodes
        return new_nodes


class CTreePyramidSampling(CSamplingMethod):
    def __init__(self, space_min, space_max):
        super(self.__class__, self).__init__(space_min, space_max)
        self.T = CTreePyramid(space_min, space_max)
        self.ndims = len(space_min)

    def reset(self):
        self.T = CTreePyramid(self.space_min, self.space_max)

    def importance_sample(self, target_d, n_samples, timeout=60, method="simple", resampling="ancestral"):
        """
        Generate n_samples using the selected importance sampling method
        :param target_d: Target distribution. Required to compute importance weights.
        :param n_samples: Desired number of samples to generate.
        :param timeout: Time-budget to generate the samples.
        :param method: Importance sampling method used. Options are: "simple", "dm" and "mixture":
            simple: Simple tree pyramid IS. Uses the simple muti-importance sampling approach by generating samples from
            all the isolated proposal distributions and computes weights with the individual distributions

            TODO: dm: Deterministic mixture tree pyramid IS. Uses the DM sampling approach by generating samples from the
            all the isolated proposal distributions but computes weights with the single distribution formed by the
            iso-weighted mixture model of all the proposals.

            TODO: mixture: Mixture tree pyramid IS. Uses the mixture sampling approach by generating samples
            from a single distribution formed by a weighted mixture of all the proposals. Importance weights are
            computed according to the same weighted mixture.

        :param resampling: Type of resampling used, options are: "none", "ancestral", "leaf", "full".
            none: No resampling. All samples drawn are kept. Outputs samples from leaves and intermediate nodes.
            ancestral: Samples from parent nodes are removed from the sample set. Outputs only samples from leaves.
            leaf: At every sampling step, all leaves are resampled. Outputs samples from leaves and intermediate nodes.
            full: Perform leaf resampling and outputs samples only from leaves.
        :return: samples and weights
        """
        assert resampling in ["leaf", "none", "ancestral", "full"], "Unknown resampling strategy"

        if method == "simple":
            samples, weights = self._importance_sample_stp(target_d, n_samples, timeout, resampling)
        elif method == "dm":
            samples, weights = self._importance_sample_stp(target_d, n_samples, timeout, resampling)
        elif method == "mixture":
            samples, weights = self._importance_sample_stp(target_d, n_samples, timeout, resampling)
        else:
            raise ValueError("Unknown method or resampling")

        return samples, weights

    def sample(self, n_samples):
        # Select n_samples leaf nodes weighted by their importance weight
        # Generate a sample from each selected leaf node
        raise NotImplementedError

    def _expand_nodes(self, particles, max_parts):
        new_particles = []
        for p in particles:
            new_parts = self.T.expand(p)
            new_particles.extend(new_parts)
            if len(new_particles) > max_parts:
                return new_particles

        return new_particles

    def _get_nsamples(self, resampling):
        res = len(self.T.leaves)
        if resampling == "none" or resampling == "leaf":
            res = len(self.T.nodes)
        return res

    def _importance_sample_stp(self, target_d, n_samples=10000, timeout=60, resampling="none"):
        """

        :param target_d:
        :param n_samples:
        :param timeout:
        :return:
        """
        elapsed_time = 0
        t_ini = time.time()

        # When the tree is created the root node is not sampled. Make sure it has one sample.
        if len(self.T.root.weight_hist) == 0:
            self.T.root.sample()
            self.T.root.weigh(target_d, self.T.root.sampler)

        while n_samples > self._get_nsamples(resampling) and elapsed_time < timeout:
            if resampling == "leaf" or resampling == "full":        # If leaf resampling is enabled
                for node in self.T.leaves:                              # For each leaf node
                    node.sample()                                       # Generate a sample
                    node.weigh(target_d, node.sampler)                  # Compute its importance weight

            lambda_hat = sorted(self.T.leaves, reverse=True)            # This is the lambda_hat set (sorted leaves)
            new_nodes = self._expand_nodes(lambda_hat, n_samples - self._get_nsamples(resampling))       # Generate the new nodes to sample

            for node in new_nodes:
                node.sample()                                           # Generate a sample for each new node,
                node.weigh(target_d, node.sampler)                      # Compute its importance weight

            elapsed_time = time.time() - t_ini

        return self._get_samples(resampling == "ancestral" or resampling == "full")

    def _sample_stpr(self, target_d, n_samples, timeout):
        raise NotImplementedError

    def _sample_dmtp(self, target_d, n_samples, timeout):
        raise NotImplementedError

    def _sample_dmtpr(self, target_d, n_samples, timeout):
        raise NotImplementedError

    def _sample_mtp(self, target_d, n_samples, timeout):
        raise NotImplementedError

    def _sample_mtpr(self, target_d, n_samples, timeout):
        raise NotImplementedError

    def _get_samples(self, only_leaves=True):
        """
        Traverse the tree nodes and return samples depending on the resampling setting
        - No resampling, return all samples
        - Ancestral or full, return samples of leaves only
        :param only_leaves: return samples only from leaves
        :return:
        """
        values_acc = t_tensor([])
        samples_acc = t_tensor([])
        for node in self.T.nodes:
            n_x = t_tensor(node.coords)
            n_w = node.weight
            if node.is_leaf() or not only_leaves and len(node.coords_hist) > 0:
                samples_acc = np.concatenate((samples_acc, n_x)) if samples_acc.size else n_x
                values_acc = np.concatenate((values_acc, n_w)) if values_acc.size else n_w

        return samples_acc.reshape(-1, self.ndims), values_acc
