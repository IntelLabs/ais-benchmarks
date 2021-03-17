
# Lu, Xiaoyu, Tom Rainforth, Yuan Zhou, Jan-Willem van de Meent, and Yee Whye Teh. “On Exploration,
# Exploitation and Learning in Adaptive Importance Sampling.” ArXiv:1810.13296 [Cs, Stat], October 31, 2018.
# http://arxiv.org/abs/1810.13296.

# Key differences with TP-AIS
# In TP-AIS there's no need to traverse the tree for sampling. Each leaf weight is stored in a list forming a
# discrete distribution to do a two step sample, select the leaf and then sample the uniform kernel.

# In TP-AIS partitioning can be limited by number of samples and depth. Every sample can be used to improve the
# proposal. No need to tune manually the ESS threshold to partition.

# TP-AIS partitions all dimensions using a TP (generalization of quadtrees, octrees etc...) instead of dimension
# by dimension sequentially. This allows for some parallelism.

# The regret analysis is very interesting and emphasizes the relevance of when to split and control the number of
# partitions. The exploration vs. exploitation discussion comments about when to split a partition (explore) or when
# not to do it (exploit) also the sigma parameter is involved with that trade-off by encouraging sampling on less
# explored subspaces.

# The problem of unlucky samples is addressed by having a factor that boosts the probability of a subspace being
# sampled. This factor depends on the number of samples that have been generated in such subspace and its initial
# value needs to be tuned

# Parameters: target ESS, sigma, boundaries

import time
import numpy as np
from numpy import array as t_tensor

from ais_benchmarks.sampling_methods.base import CMixtureISSamplingMethod
from ais_benchmarks.distributions import CMixtureModel
from ais_benchmarks.distributions import CMultivariateUniform

from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib import cm


class CHiDaiseeNode:
    def __init__(self, idx, center, radius):
        self.center = center
        self.radius = radius
        self.q = 0
        self.idx = idx
        self.left = None
        self.right = None
        self.parent = None
        self.sampler = CMultivariateUniform({"center": center, "radius": radius})
        self.samples = np.array([])
        self.weights = np.array([])
        self.N = 0
        self.ESS = 0

    def prob(self, x):
        return self.sampler.prob(x)

    def log_prob(self, x):
        return self.sampler.log_prob(x)

    def add_sample(self, x, w):
        self.samples = np.concatenate((self.samples, x)) if self.samples.size else x
        self.weights = np.concatenate((self.weights, w)) if self.weights.size else w
        self.N = len(self.samples)
        self.ESS = (np.sum(self.weights) * np.sum(self.weights)) / np.sum(self.weights * self.weights)

    def sample(self):
        return self.sampler.sample()

    def tree_to_str(self):
        return

    def __str__(self):
        if self.N > 0:
            return "[ idx:" + str(self.idx) + " , c:" + str(self.center) + " , r:" + str(self.radius) + \
                   " , q: %8.6f" % self.q + " , ESS: %5.3f" % (self.ESS / self.N) + " , N:" + str(self.N) + " ]"

        return "[ idx:" + str(self.idx) + " , c:" + str(self.center) + " , r:" + str(self.radius) + \
               " , q: %8.6f" % self.q + " , ESS: %5.3f" % self.ESS + " , N:" + str(self.N) + " ]"

    def __repr__(self):
        return self.__str__()


class CHiDaiseeTree:
    def __init__(self, space_min, space_max):
        self.dims = len(space_min)
        self.space_min = space_min
        self.space_max = space_max

        center = (self.space_max + self.space_min) / 2
        radius = (self.space_max - self.space_min) / 2
        self.nodes = [CHiDaiseeNode(0, center, radius)]
        self.leaves = [0]

    def split(self, node_id, target_d):
        n = self.nodes[node_id]
        new_rad = np.copy(n.radius)
        # Select a dimension to split
        sel_dim = np.random.randint(low=0, high=self.dims)
        new_rad[sel_dim] /= 2
        c_left = np.copy(n.center)
        c_right = np.copy(n.center)
        c_left[sel_dim] -= new_rad[sel_dim]
        c_right[sel_dim] += new_rad[sel_dim]
        n_left = CHiDaiseeNode(len(self.nodes), c_left, new_rad)
        n_right = CHiDaiseeNode(len(self.nodes) + 1, c_right, new_rad)
        n_left.parent = node_id
        n_right.parent = node_id
        n_left.q = n.q / 2
        n_right.q = n.q / 2

        n.left = n_left
        n.right = n_right

        self.nodes.append(n_left)
        self.nodes.append(n_right)

        # Push the samples from the parent to the children. Because the children have different
        # proposal distributions we have to recompute the weights of existing samples
        for x in n.samples:
            if x[sel_dim] < n_left.center[sel_dim] + n_left.radius[sel_dim]:
                if n_left.prob(x) <= 0:
                    raise ValueError("pl(x) = %f ; pr(x) = %f " % (n_left.prob(x), n_right.prob(x)))
                w = target_d.prob(x) / n_left.prob(x)
                n_left.add_sample(np.array([x]), w)
            else:
                if n_right.prob(x) <= 0:
                    raise ValueError("pl(x) = %f ; pr(x) = %f " % (n_left.prob(x), n_right.prob(x)))
                w = target_d.prob(x) / n_right.prob(x)
                n_right.add_sample(np.array([x]), w)

        # Remove parent from the leaf list and add children
        self.leaves.remove(node_id)
        self.leaves.append(n_left.idx)
        self.leaves.append(n_right.idx)

        # Clear the samples from the parent node
        n.samples = None

    def tree_to_str(self, n, depth=0):
        res = ""
        # If is a leaf. Print the leaf
        if n.left is None and n.right is None:
            return str(self.nodes[n.idx]) + "\n"

        res += "| [idx: %d , q:%f]" % (n.idx, n.q) + "\n"
        res += "|" + depth * "--" + "--| %d " % depth + self.tree_to_str(self.nodes[n.left.idx], depth+1)
        res += "|" + depth * "--" + "--| %d " % depth + self.tree_to_str(self.nodes[n.right.idx], depth+1)
        return res

    def __str__(self):
        return self.tree_to_str(self.nodes[0])

    def __repr__(self):
        return self.__str__()


class CHiDaiseeSampling(CMixtureISSamplingMethod):
    def __init__(self, params):
        """
        Initialize the Tree Pyramid sampling with the specific parameters
        :param params: Dictionary with sampling algorithm specific parameters
            space_max: Upper space domain values
            space_min: Lower space domain values
            debug: Debug mode flag
            target_ess: Target ESS used for the tree splitting criteria.
            tau: Initial exploration weight. It is divided by half after every split. Also referred to as the
                 parameter that controls the optimism boost sigma_a
            n_min: Minimum number of samples for splitting a node.
        """
        super(self.__class__, self).__init__(params)
        self.tau = params["tau"]
        assert self.tau > 0, "Sigma value must be positive."
        self.sigma = np.sqrt(4.14 * np.log2(2*np.e)) * self.tau  # NOTE: From Theorem 1.

        self.n_min = params["n_min"]
        assert self.n_min > 0, "Nmin value must be positive."

        self.ess_target = params["target_ess"]
        assert 0 < self.ess_target < 1, "Invalid target ESS value. Must be (0,1]"

        self.debug = params["debug"]

        self.reset()

    def reset(self):
        super(CHiDaiseeSampling, self).reset()

        if self.debug:
            print(self.__class__.__name__, "== Debug ==> ", "reset")

        super(self.__class__, self).reset()
        self.T = CHiDaiseeTree(self.space_min, self.space_max)

        center = (self.space_max + self.space_min) / 2
        radius = (self.space_max - self.space_min) / 2
        self.model = CMultivariateUniform({"center": center, "radius": radius})

    def importance_sample(self, target_d, n_samples, timeout=60):
        if self.debug:
            print(self.__class__.__name__, "== Debug ==> ", "importance_sample(n_samples=%d)" % n_samples)

        for t in range(len(self.samples), n_samples):
            # Initialize traversal path and node to the root. Line 3.
            P = [0]
            n_id = 0

            if self.debug:
                print(self.__class__.__name__, "== Debug ==> ", "-- Select node")

            # Select the leaf node to sample from. Lines 4 to 9
            while n_id not in self.T.leaves:
                if self.debug:
                    print(self.__class__.__name__, "== Debug ==> ", "    | Choose a children")
                # Line 5: Append the node sampled to the sequence
                P.append(n_id)

                # Lines 6 to 8 Select which children to use
                r = np.random.rand()
                qil = self.T.nodes[n_id].left.q
                qir = self.T.nodes[n_id].right.q
                q_left = qil / (qir + qil)
                n_id = self.T.nodes[n_id].left.idx if r < q_left else self.T.nodes[n_id].right.idx
                if self.debug:
                    print(self.__class__.__name__, "== Debug ==> ", "    | Choosen %d" % n_id)

            if self.debug:
                print(self.__class__.__name__, "== Debug ==> ", "    | Selected %d" % n_id)

            # Sample the selected node. Line 10.
            n = self.T.nodes[n_id]
            x_i = n.sample()
            self._num_q_samples += 1

            # Compute the sample weight. Line 10. Partially in log form for numerical stability.
            assert n.prob(x_i) > 0
            w_i = target_d.prob(x_i) / n.prob(x_i)
            self._num_q_evals += 1
            self._num_pi_evals += 1

            # Add the sample to the leaf node and update ESS and N. Line 12
            n.add_sample(x_i, w_i)
            if self.debug:
                print(self.__class__.__name__, "== Debug ==> ", "    | Added sample x:%5s w:%5s" % (str(x_i), str(w_i)))
                print(self.__class__.__name__, "== Debug ==> ", "    | Node %d now has %d samples with ESS: %5.3f" % (n_id, n.N, n.ESS))
                print(self.__class__.__name__, "== Debug ==> ", "    | and weights %s" % str(n.weights.flatten()))

            # Update the leaf and ancestors proposal expected evidence values. Line 11
            # Because the traversal trajectory is stored in P. The last node is a leaf node and the remaining nodes
            # in the list are ancestors of P.
            # qi computed using eq.8 for leaves. Which is a MC estimate of the evidence in the subspace.

            # Compute the optimism boost (sigma_a) according to the paper description in Theorem 1.
            sigma_at = self.sigma * np.sqrt(np.log(t) / n.N)

            # Compute the denominator of eq.8
            total_ev = 0
            for leaf_idx in self.T.leaves:
                if self.T.nodes[leaf_idx].N > 0:
                    sigma_bt = self.sigma * np.sqrt(np.log(t) / self.T.nodes[leaf_idx].N)
                    Zb = np.mean(self.T.nodes[leaf_idx].weights)
                    total_ev += Zb + sigma_bt

            Za = np.mean(n.weights)
            n.q = (Za + sigma_at) / total_ev

            # qi computed using eq.19 for ancestors. Line 11
            for parent_n in reversed(P[:-1]):
                self.T.nodes[parent_n].q = self.T.nodes[parent_n].left.q + self.T.nodes[parent_n].right.q

            # Determine whether to split the leaf node. Lines 12 to 15
            if n.N >= self.n_min and n.ESS < self.ess_target * n.N:
                if self.debug:
                    print(self.__class__.__name__, "== Debug ==> ", "xx Split node %d" % n_id)
                # Account for the evaluations required for the existing samples after a node split
                self._num_q_evals += n.N
                self._num_pi_evals += n.N
                self.T.split(n_id, target_d)

            if self.debug:
                print(self.__class__.__name__, "== Debug ==> ", "    | Current tree: \n", self.T.__repr__())

        self._update_model()
        return self._get_samples()

    def _self_normalize(self):
        # Compute the sum of all the weights in the tree
        # Normalize all weights in the tree
        raise NotImplementedError

    def _get_weights_nodes(self):
        centers = list()
        radii = list()
        weights = list()
        for n_idx in self.T.leaves:
            centers.append(self.T.nodes[n_idx].center)
            radii.append(self.T.nodes[n_idx].radius)
            weights.append(np.mean(self.T.nodes[n_idx].weights))
        return centers, radii, weights

    def _get_samples(self):
        samples = list()
        weights = list()
        for n_idx in self.T.leaves:
            samples.extend(self.T.nodes[n_idx].samples)
            weights.extend(self.T.nodes[n_idx].weights)

        self.samples = np.array(samples)
        self.weights = np.array(weights)
        return self.samples, self.weights

    def _update_model(self):
        centers, radii, weights = self._get_weights_nodes()
        models = []
        for c, r in zip(centers, radii):
            models.append(CMultivariateUniform({"center": np.array(c), "radius": np.array(r)}))

        self.model = CMixtureModel(models, weights)

    @staticmethod
    def draw_2d_tree(T, ax, facecolor=(1, 1, 1), edgecolor=(0, 0, 0), alpha=1.0):
        res = []
        for l in T.leaves:
            c = T.nodes[l].center
            r = T.nodes[l].radius
            q = T.nodes[l].q
            rect = Rectangle((c[0] - r[0], c[1] - r[1]), 2*r[0], 2*r[1])

            # Create patch collection with specified colour/alpha
            pc = PatchCollection([rect], facecolor=cm.hot(1-q), alpha=alpha, edgecolor=edgecolor)
            # if T.nodes[l].N > 0:
            #     pc = PatchCollection([rect], facecolor=cm.hot(T.nodes[l].ESS / T.nodes[l].N), alpha=alpha, edgecolor=edgecolor)
            # else:
            #     pc = PatchCollection([rect], facecolor=cm.hot(0), alpha=alpha, edgecolor=edgecolor)
            ax.add_collection(pc)
            res.append(pc)

        return res

    def draw2d(self, ax):
        res = self.draw_2d_tree(self.T, ax)
        return res
