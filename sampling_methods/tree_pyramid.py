import time
import numpy as np
from numpy import array as t_tensor
import itertools

from sampling_methods.base import CSamplingMethod
from distributions.CMultivariateNormal import CMultivariateNormal
from distributions.CMultivariateUniform import CMultivariateUniform
from utils.plot_utils import plot_tpyramid_area
from utils.plot_utils import plot_pdf
from utils.plot_utils import plot_pdf2d
import matplotlib.cm as cm


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
        self.coords = t_tensor([[0] * len(center)]) # n_x
        self.leaf_idx = leaf_idx
        self.node_idx = node_idx
        self.level = level
        self.value = self.weight * self.radius ** len(center)  # Importance volume used to sort nodes
        self.coords_hist = []
        self.weight_hist = []

        """
        Sampling kernel distribution. This shapes the proposal distribution represented by the tree pyramid.
        """
        if kernel == "haar":
            self.sampler = CMultivariateUniform(self.center, self.radius)
        elif kernel == "normal":
            self.sampler = CMultivariateNormal(self.center, np.diag(t_tensor([self.radius/4] * len(self.center))))
        else:
            raise ValueError("Unknown kernel type. Must be 'normal' or 'haar'")

    def sample(self):
        """
        This is the Monte Carlo portion of the Quasi-Monte Carlo approach. The sample location (n_x) is obtained
        by a proposal distribution parameterized by the node center (n_c) and radius (n_r).
        """
        self.coords = self.sampler.sample()
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
    def __init__(self, dmin, dmax, kernel):
        self.root = CTreePyramidNode(center=(dmin + dmax) / 2, radius=np.max((dmax - dmin) / 2),
                                     leaf_idx=0, node_idx=0, level=0, kernel=kernel)
        self.nodes = [self.root]
        self.leaves = [self.root]
        self.ndims = len(dmin)
        self.kernel = kernel

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
                                              level=node.level + 1, kernel=self.kernel))

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
    def __init__(self, space_min, space_max, params):
        """
        Initialize the Tree Pyramid sampling with the specific parameters
        :param space_min: Lower space domain values
        :param space_max: Upper space domain values
        :param params: Dictionary with sampling algorithm specific parameters
            method: Importance sampling method used. Options are: "simple", "dm" and "mixture":
                - simple: Simple tree pyramid IS. Uses the simple muti-importance sampling approach by generating
                samples from all the isolated proposal distributions and computes weights with the individual
                distributions.

                - TODO: dm: Deterministic mixture tree pyramid IS. Uses the DM sampling approach by generating samples
                from the all the isolated proposal distributions but computes weights with the single distribution
                formed by the iso-weighted mixture model of all the proposals.

                - TODO: mixture: Mixture tree pyramid IS. Uses the mixture sampling approach by generating samples
                from a single distribution formed by a weighted mixture of all the proposals. Importance weights are
                computed according to the same weighted mixture.

            resampling: Type of resampling used, options are: "none", "ancestral", "leaf", "full".
                - none: No resampling. All samples drawn are kept. Outputs samples from leaves and intermediate nodes.
                - ancestral: Samples from parent nodes are removed from the sample set. Output only samples from leaves.
                - leaf: At every sampling step, all leaves are resampled. Output from leaves and intermediate nodes.
                - full: Perform leaf resampling and outputs samples only from leaves.

            kernel: Kernel type used for the proposal distributions represented by the tree nodes (i.e. subspaces)
                - haar: Uniform distribution with min=node.center-node.radius and max=node.center-node.radius.
                - normal: Normal distribution with mean=node.center and std=node.radius.
        """
        super(self.__class__, self).__init__(space_min, space_max)
        self.method = params["method"]
        assert self.method in ["simple", "dm", "mixture"], "Invalid method."

        self.resampling = params["resampling"]
        assert self.resampling in ["leaf", "none", "ancestral", "full"], "Invalid resampling strategy"

        self.kernel = params["kernel"]
        assert self.kernel in ["normal", "haar"], "Invalid kernel type."

        self.T = CTreePyramid(space_min, space_max, kernel=self.kernel)
        self.ndims = len(space_min)

    def reset(self):
        super(self.__class__, self).reset()
        self.T = CTreePyramid(self.space_min, self.space_max, kernel=self.kernel)

    def importance_sample(self, target_d, n_samples, timeout=60):
        """
        Obtain n_samples importance samples with their importance weights
        :param target_d: target distribution. Must implement the target_d.prob(samples) method that returns the prob for
        a batch of samples.
        :param n_samples: total number of samples to obtain, considering the samples already generated from prior calls
        :param timeout: maximum time allowed to obtain the required number of samples
        :return: samples and weights
        """

        assert self.resampling in ["leaf", "none", "ancestral", "full"], "Unknown resampling strategy"

        if self.method == "simple":
            samples, weights = self._importance_sample_stp(target_d, n_samples, timeout)
        elif self.method == "dm":
            samples, weights = self._importance_sample_dmtp(target_d, n_samples, timeout)
        elif self.method == "mixture":
            samples, weights = self._importance_sample_mtp(target_d, n_samples, timeout)
        else:
            raise ValueError("Unknown method or resampling")

        return samples, weights

    def prob(self, s):
        prob = 0
        for n in self.T.leaves:
            prob = prob + n.sampler.prob(s) * n.weight# if self.kernel == "haar" else prob + n.sampler.prob(s)
            self._num_q_evals += 1
        return prob if self.kernel == "haar" else prob / len(self.T.leaves)

    def logprob(self, s):
        return self.prob(s)

    def sample(self, n_samples):
        # TODO: Implement the sample method
        # Select n_samples leaf nodes weighted by their importance weight
        # Generate a sample from each selected leaf node
        raise NotImplementedError

    def draw(self, ax):
        res = []
        if self.ndims == 1:
            res = plot_tpyramid_area(ax, self.T, label="$w(x) = \pi(x)/q(x)$")
            res.extend(plot_pdf(ax, self, self.space_min, self.space_max, resolution=0.01,
                                options="-r", alpha=1.0, label="$q(x)$"))
        elif self.ndims == 2:
            res.append(plot_pdf2d(ax, self, self.space_min, self.space_max, alpha=0.5, resolution=0.02, colormap=cm.viridis, label="$q(x)$"))
        return res

    def _expand_nodes(self, particles, max_parts):
        new_particles = []
        for p in particles:
            new_parts = self.T.expand(p)
            new_particles.extend(new_parts)
            if len(new_particles) > max_parts:
                return new_particles

        return new_particles

    def get_acceptance_rate(self):
        return self._get_nsamples() / self._num_q_samples

    def _get_nsamples(self):
        res = len(self.T.leaves)
        if self.resampling == "none" or self.resampling == "leaf":
            res = len(self.T.nodes)
        return res

    def _importance_sample_stp(self, target_d, n_samples=10000, timeout=60):
        elapsed_time = 0
        t_ini = time.time()

        # When the tree is created the root node is not sampled. Make sure it has one sample.
        if len(self.T.root.weight_hist) == 0:
            self.T.root.sample()
            self._num_q_samples += 1
            self.T.root.weigh(target_d, self.T.root.sampler)

        while n_samples > self._get_nsamples() and elapsed_time < timeout:
            if self.resampling == "leaf" or self.resampling == "full":  # If leaf resampling is enabled
                for node in self.T.leaves:                              # For each leaf node
                    node.sample()                                       # Generate a sample
                    self._num_q_samples += 1                            # Count the sample operation
                    node.weigh(target_d, node.sampler)                  # Compute its importance weight
                    self._num_pi_evals += 1                             # Count the evaluation operation
                    self._num_q_evals += 1                              # Count the evaluation operation

            lambda_hat = sorted(self.T.leaves, reverse=True)            # This is the lambda_hat set (sorted leaves)
            new_nodes = self._expand_nodes(lambda_hat, n_samples - self._get_nsamples())  # Generate the new nodes to sample

            for node in new_nodes:
                node.sample()                                           # Generate a sample for each new node,
                self._num_q_samples += 1  # Count the sample operation
                node.weigh(target_d, node.sampler)                      # Compute its importance weight
                self._num_pi_evals += 1  # Count the evaluation operation
                self._num_q_evals += 1  # Count the evaluation operation

            elapsed_time = time.time() - t_ini

        return self._get_samples()

    def _importance_sample_dmtp(self, target_d, n_samples, timeout):
        raise NotImplementedError

    def _importance_sample_mtp(self, target_d, n_samples, timeout):
        raise NotImplementedError

    def _get_samples(self):
        """
        Traverse the tree nodes and return samples depending on the resampling setting
        - No resampling, return all samples
        - Ancestral or full, return samples of leaves only
        :return: samples, importance_weights
        """
        only_leaves = self.resampling == "ancestral" or self.resampling == "full"
        values_acc = t_tensor([])
        samples_acc = t_tensor([])
        for node in self.T.nodes:
            n_x = t_tensor(node.coords)
            n_w = node.weight
            if node.is_leaf() or not only_leaves and len(node.coords_hist) > 0:
                samples_acc = np.concatenate((samples_acc, n_x)) if samples_acc.size else n_x
                values_acc = np.concatenate((values_acc, n_w)) if values_acc.size else n_w

        return samples_acc.reshape(-1, self.ndims), values_acc
