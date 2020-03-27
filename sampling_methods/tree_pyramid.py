import time
import numpy as np
from numpy import array as t_tensor
import itertools
import matplotlib.patches as patches

from sampling_methods.base import CMixtureISSamplingMethod
from distributions.parametric.CMultivariateNormal import CMultivariateNormal
from distributions.parametric.CMultivariateUniform import CMultivariateUniform
from utils.plot_utils import plot_tpyramid_area
from utils.plot_utils import plot_pdf
from utils.plot_utils import plot_pdf2d
from distributions.mixture.CMixtureModel import CMixtureModel


class CTreePyramidNode:
    def __init__(self, center, radius, node_idx, leaf_idx, level, kernel="haar", bw_div=4, weight=t_tensor(0)):
        """
        A node of a tree pyramid contains information about its location
        :param center: location of the center of the node (In the paper: n_c)
        :param radius: radius of the node (In the paper: n_r)
        :param leaf_idx: Index of the node in the tree leaf list. If the node is not a leaf this index is -1
        :param node_idx: Index of the node in the tree node list
        :param level: Distance of the node to the root. Must be consistent with the children and parents.
        :param kernel: Kernel type to be used for this node sampler
        :param bw_div: The kernel bandwidth is computed as radius/bw_div. This param slightly tunes the amount of
                       smoothing because it is automatically adapted by being proportional to the node radii.
        """
        self.center = t_tensor(center)               # n_c
        self.radius = t_tensor(radius)               # n_r
        self.children = [None] * (2**len(center))    # n_s
        self.weight = weight                         # n_w
        self.coords = t_tensor([[0] * len(center)])  # n_x
        self.leaf_idx = leaf_idx
        self.node_idx = node_idx
        self.level = level
        self.value = self.weight * self.radius ** len(center)  # Importance volume used to sort nodes
        self.coords_hist = []
        self.weight_hist = []
        self.bw_div = bw_div

        """
        Sampling kernel distribution. This shapes the proposal distribution represented by the tree pyramid.
        """
        if kernel == "haar":
            self.sampler = CMultivariateUniform({"center": self.center, "radius": self.radius})
        elif kernel == "normal":
            self.sampler = CMultivariateNormal({"mean": self.center,
                                                "sigma": np.diag(t_tensor([self.radius/self.bw_div] * len(self.center)))
                                                })
        else:
            raise ValueError("Unknown kernel type. Must be 'normal' or 'haar'")

    def sample(self, sampler):
        """
        This is the Monte Carlo portion of the Quasi-Monte Carlo approach. The sample location (n_x) is obtained
        by a proposal distribution parameterized by the node center (n_c) and radius (n_r).
        """
        self.coords = sampler.sample(1)
        self.coords_hist.append(self.coords)

    def weigh(self, target_d, importance_d, target_f=lambda x: 1):
        """
        This is the importance sampling part where the weight of the sample is updated.

        :param target_d: Target distribution to compute the importance weight. In the paper: \pi(x)
        :param importance_d: Importance distribution used to generate the sample. In the paper: q(x)
        :param target_f: Target f(x) used for importance sampling.
        """
        # TODO: There is a problem here for the DM weights that try to compute a weight to the particle using
        # the mixture distribution. The weights are initially zero and if they are used to compute the sample
        # weight there is a cirlular dependency on this calculation. In the case where all the parent nodes are
        # sub-divided at the same iteration, all the new leaves have initial weight of zero until it is computed.
        # The problem is that all leaves weights are zero and the importance is zero as well.
        importance = importance_d.prob(self.coords)
        self.weight = (target_f(self.coords) * target_d.prob(self.coords)) / importance
        # self.weight = (target_f(self.center) * target_d.prob(self.center)) / importance_d.prob(self.center)
        self.value = self.weight * (self.radius ** len(self.center))
        self.weight_hist.append(self.weight)

    def get_child(self, val):
        """
        :param val: Value used to determine the children that contains it
        :return: children node that contains val
        """
        for c in self.children:
            min_val = c.center - c.radius
            max_val = c.center + c.radius
            if np.all(min_val <= val) and np.all(val <= max_val):
                return c
        raise ValueError("Children value not found.")

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
        self.root = CTreePyramidNode(center=(dmin + dmax) / 2, radius=np.array([np.max((dmax - dmin) / 2)]),
                                     leaf_idx=0, node_idx=0, level=0, kernel=kernel)
        self.ndims = len(dmin)
        self.nodes = [self.root]
        self.leaves = [self.root]
        self.centers = np.array([self.root.center])
        self.radii = np.array([self.root.radius])
        self.samples = np.zeros((1, self.ndims))
        self.weights = np.array([0])
        self.leaves_idx = [True]

        self.kernel = kernel

        """
        Resampling distribution. This is used to resample the leaves by sampling from a prototypical distribution
        and scaling the sample depending on the leaf location and radius.
        """
        if kernel == "haar":
            self.sampler = CMultivariateUniform({"center": np.zeros(self.ndims),
                                                 "radius": np.ones(self.ndims)*0.5})
        elif kernel == "normal":
            self.sampler = CMultivariateNormal({"mean": np.zeros(self.ndims),
                                                "sigma": np.diag(np.ones(self.ndims))*0.5})
        else:
            raise ValueError("Unknown kernel type. Must be 'normal' or 'haar'")

    def find(self, val, prev=None):
        """
        :param val: value to find the node that contains it
        :param prev: node that represents the subspace to search
        :return: node that contains val
        """
        if prev is None:
            prev = self.root

        if prev is not None and prev.is_leaf():
            return prev
        else:
            return self.find(val, prev=prev.get_child(val))

    def expand(self, node):
        assert node.is_leaf(), "Non leaf node expansion is not allowed: " + repr(node)

        ################################################
        # Compute new node centers and radii
        ################################################
        center_coeffs = itertools.product([-1, 1], repeat=len(node.center))

        # TODO: Vectorize this operation
        p_centers = []
        for N, coeff in enumerate(center_coeffs):
            offset = (node.radius / 2) * t_tensor(coeff)
            p_centers.append(node.center + offset)

        new_radius = node.radius / 2
        ################################################
        ################################################

        ################################################
        # Create the new child nodes and update the tree
        ################################################
        # Create the new child nodes that are initialized with their fair share of their parent weight, this
        # addresses the problem of all the weights of the leaf nodes being zero.
        new_nodes = []
        for i, center in enumerate(p_centers):
            new_nodes.append(CTreePyramidNode(center=center, radius=new_radius, weight=node.weight / (N+1),
                                              leaf_idx=len(self.leaves) + i - 1, node_idx=len(self.nodes) + i,
                                              level=node.level + 1, kernel=self.kernel))

        # The expanding node is not a leaf anymore
        self.leaves_idx[node.node_idx] = False

        # The first new node replaces the expanding leaf node in the leaves list
        new_nodes[0].leaf_idx = node.leaf_idx
        self.leaves[node.leaf_idx] = new_nodes[0]
        self.nodes.append(new_nodes[0])
        # Add the node center and radius to the cached list
        self.centers = np.concatenate((self.centers, t_tensor([new_nodes[0].center])))
        self.radii = np.concatenate((self.radii, t_tensor([new_nodes[0].radius])))
        self.samples = np.concatenate((self.samples, np.zeros((1, self.ndims))))
        self.weights = np.concatenate((self.weights, np.zeros(1)))
        self.leaves_idx.append(True)

        node.children[0] = new_nodes[0]
        node.leaf_idx = -1

        # The rest get inserted into the tree
        for i, new_n in enumerate(new_nodes[1:]):
            node.children[i+1] = new_n
            new_n.leaf_idx = len(self.leaves)
            self.leaves.append(new_n)
            self.nodes.append(new_n)
            # Add the node center and radius to the cached list
            self.centers = np.concatenate((self.centers, t_tensor([new_n.center])))
            self.radii = np.concatenate((self.radii, t_tensor([new_n.radius])))
            self.samples = np.concatenate((self.samples, np.zeros((1, self.ndims))))
            self.weights = np.concatenate((self.weights, np.zeros(1)))
            self.leaves_idx.append(True)
        ################################################
        ################################################

        # Return the newly created nodes
        return new_nodes

    @staticmethod
    def draw_node(axis, x, y, label="x", color="b"):
        axis.add_patch(patches.Circle((x, y), 0.1, facecolor="w", linewidth=1, edgecolor=color))
        axis.annotate(label, xy=(x-0.2, y-0.3), fontsize=18, ha="center", va="center")
        axis.plot(x, y)

    @staticmethod
    def plot_node(axis, node, x, y, names=True):
        plot_span = 10
        if node.is_leaf():
            if names:
                CTreePyramid.draw_node(axis, x, y, "$\lambda_{%d}$" % node.leaf_idx, color="r")
            else:
                CTreePyramid.draw_node(axis, x, y, "", color="r")
        else:
            CTreePyramid.draw_node(axis, x, y, "", color="b")
            for idx,ch in enumerate(node.children):
                nodes_in_level = 2 ** (len(node.coords) * ch.level)
                if nodes_in_level > 1:
                    increment = plot_span / nodes_in_level
                else:
                    increment = 0
                x_ch = x + increment * idx - (plot_span/nodes_in_level) / 2
                y_ch = -ch.level
                axis.arrow(x, y, x_ch - x, y_ch - y, alpha=0.2, zorder=0)
                CTreePyramid.plot_node(axis, ch, x_ch, y_ch, names)

    def plot(self, axis, names=True):
        axis.cla()
        self.plot_node(axis, self.root, 0, 0, names)


class CTreePyramidSampling(CMixtureISSamplingMethod):
    def __init__(self, params):
        """
        Initialize the Tree Pyramid sampling with the specific parameters
        :param space_min: Lower space domain values
        :param space_max: Upper space domain values
        :param params: Dictionary with sampling algorithm specific parameters
            method: Importance sampling method used. Options are: "simple", "dm" and "mixture":
                - simple: Simple tree pyramid IS. Uses the simple muti-importance sampling approach by generating
                samples from all the isolated proposal distributions and computes weights with the individual
                distributions.

                - dm: Deterministic mixture tree pyramid IS. Uses the DM sampling approach by generating samples
                from the all the isolated proposal distributions but computes weights with the single distribution
                formed by the iso-weighted mixture model of all the proposals.

                - mixture: Mixture tree pyramid IS. Uses the mixture sampling approach by generating samples
                from a single distribution formed by a weighted mixture of all the proposals. Importance weights are
                computed according to the same weighted mixture.

            resampling: Type of resampling used, options are: "none", "leaf".
                - none: No resampling. All samples drawn are kept. Outputs samples from leaves and intermediate nodes.
                - leaf: At every sampling step, all leaves are resampled. Output from leaves and intermediate nodes.

            kernel: Kernel type used for the proposal distributions represented by the tree nodes (i.e. subspaces)
                - haar: Uniform distribution with min=node.center-node.radius and max=node.center-node.radius.
                - normal: Normal distribution with mean=node.center and std=node.radius.
        """
        super(self.__class__, self).__init__(params)
        self.method = params["method"]
        assert self.method in ["simple", "dm", "mixture"], "Invalid method."

        self.resampling = params["resampling"]
        assert self.resampling in ["leaf", "none"], "Invalid resampling strategy"

        self.kernel = params["kernel"]
        assert self.kernel in ["normal", "haar"], "Invalid kernel type."

        self.T = CTreePyramid(self.space_min, self.space_max, kernel=self.kernel)

    def reset(self):
        super(self.__class__, self).reset()
        self.T = CTreePyramid(self.space_min, self.space_max, kernel=self.kernel)

    def find_unique(self, samples):
        """
        Obtains the list of unique nodes that contain the samples passed in the samples parameter. Returns also a list
        with the number of samples that belong to each node of the returned list of nodes.

        :param samples: Samples to get the nodes.
        :return: (nodes, freq) nodes: List of unique nodes that contain the samples.
                               freq: List with the number of samples in each node
        """
        nodes = []
        freqs = []
        for s in samples:
            node = self.T.find(s)
            if node in nodes:
                freqs[nodes.index(node)] += 1
            else:
                nodes.append(node)
                freqs.append(1)

        return nodes, freqs

    def importance_sample(self, target_d, n_samples, timeout=60):
        """
        Obtain n_samples importance samples with their importance weights
        :param target_d: target distribution. Must implement the target_d.prob(samples) method that returns the prob for
        a batch of samples.
        :param n_samples: total number of samples to obtain, considering the samples already generated from prior calls
        :param timeout: maximum time allowed to obtain the required number of samples
        :return: samples and weights
        """

        assert self.resampling in ["leaf", "none"], "Unknown resampling strategy"
        assert self.method in ["simple", "dm", "mixture"], "Unknown method strategy"

        elapsed_time = 0
        t_ini = time.time()

        # When the tree is created the root node is not sampled. Make sure it has one sample.
        if len(self.T.root.weight_hist) == 0:
            self.T.root.sample(self.T.root.sampler)
            self.T.samples[self.T.root.node_idx] = self.T.root.coords
            self._num_q_samples += 1
            self.T.root.weigh(target_d, self.T.root.sampler)
            self.T.weights[self.T.root.node_idx] = self.T.root.weight
            self._update_model()

        while n_samples > self._get_nsamples() and elapsed_time < timeout:
            if self.method == "mixture":
                # Generate new samples from the proposal. The number of samples to generate depends on the nodes
                # that will be expanded. Each node generates 2^k samples. Therefore to obtain N samples we have to
                # split X nodes. N = X*2^k, X = N / 2^k

                n_samples_to_get = n_samples - self._get_nsamples()
                nodes_to_expand = np.int(np.ceil(n_samples_to_get / 2**self.ndims))
                samples = self.sample(nodes_to_expand)

                # Find the unique nodes for the sample
                nodes, freqs = self.find_unique(np.clip(samples, self.space_min, self.space_max))

                # Expand them
                new_nodes = self._expand_nodes(nodes, n_samples - self._get_nsamples())
            else:
                lambda_hat = sorted(self.T.leaves, reverse=True)            # This is the lambda_hat set (sorted leaves)
                new_nodes = self._expand_nodes(lambda_hat, n_samples - self._get_nsamples())  # Generate the new nodes to sample

            for node in new_nodes:
                if self.method == "simple" or self.method == "dm":
                    node.sample(node.sampler)
                elif self.method == "mixture":
                    node.sample(self.model)

                self.T.samples[node.node_idx] = node.coords
                self._num_q_samples += 1  # Count the sample operation

                if self.method == "simple":
                    node.weigh(target_d, node.sampler)
                elif self.method == "mixture" or self.method == "dm":
                    node.weigh(target_d, self)
                self._num_pi_evals += 1  # Count the evaluation operation
                self._num_q_evals += 1  # Count the evaluation operation
                self.T.weights[node.node_idx] = node.weight

            # If leaf resampling is enabled
            if self.resampling == "leaf":
                re_samples = self.T.sampler.sample(len(self.T.leaves))
                centers = self.T.centers[self.T.leaves_idx]
                radii = self.T.radii[self.T.leaves_idx].reshape(len(re_samples), 1)
                samples = re_samples * radii + centers
                if self.kernel == "haar":
                    # Because in the haar case the importance probability depends on the node radii, we can omit
                    # calling self.prob() (which computes computing q(x)) and do it in a vectorized form as shown below.
                    importance_probs = 1 / ((2*radii)**self.T.ndims)
                elif self.kernel == "normal":
                    importance_probs = self.T.sampler.prob(re_samples)
                self.T.samples[self.T.leaves_idx] = samples
                probs = target_d.prob(samples)
                self.T.weights[self.T.leaves_idx] = probs.reshape(-1) / importance_probs.reshape(-1)
                self._num_pi_evals += len(samples)
                self._num_q_evals += len(samples)
                self._num_q_samples += len(samples)

                # Self-normalization of importance weights.
                self._self_normalize()
                for node in self.T.leaves:
                    node.weight = self.T.weights[node.node_idx]
                    node.value = node.weight * ((2*node.radius) ** len(node.center))

            # Force an update of the tree parameterized distributions after the updated weights
            # Update model internally calls self_normalize. So weights are always self-normalized when the model is
            # updated.
            self._update_model()

            elapsed_time = time.time() - t_ini

        return self._get_samples()

    def draw(self, ax):
        res = []
        if self.ndims == 1:
            res = plot_tpyramid_area(ax, self.T, label="TP-AIS $w(x) = \pi(x)/q(x)$")
            res.extend(plot_pdf(ax, self, self.space_min, self.space_max, resolution=0.01,
                                options="-g", alpha=1.0, label="TP-AIS $q(x)$"))

            if self.kernel == "normal":
                for n in self.T.leaves:
                    res.extend(plot_pdf(ax, n.sampler, self.space_min, self.space_max, scale=n.weight,
                                        resolution=0.01, options="--r", alpha=0.5))

        elif self.ndims == 2:
            res.extend(plot_pdf2d(ax, self, self.space_min, self.space_max, alpha=0.5, resolution=0.02, label="TP-AIS $q(x)$"))
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

    def get_NESS(self):
        samples, weights = self._get_samples()
        ESS = 1 / np.sum(weights*weights)
        return ESS / len(samples)

    def _get_nsamples(self):
        res = len(self.T.leaves)
        return res

    def _get_samples(self):
        return self.T.samples[self.T.leaves_idx], self.T.weights[self.T.leaves_idx]

    def _self_normalize(self):
        self.T.weights[self.T.leaves_idx] = self.T.weights[self.T.leaves_idx] / np.sum(self.T.weights[self.T.leaves_idx])

    def _update_model(self):
        weights = self.T.weights[self.T.leaves_idx]
        centers = self.T.centers[self.T.leaves_idx]
        radii = self.T.radii[self.T.leaves_idx]
        models = []
        for c, r in zip(centers, radii):
            if self.kernel == "haar":
                models.append(CMultivariateUniform({"center": c, "radius": r}))

            elif self.kernel == "normal":
                models.append(CMultivariateNormal({"mean": c,
                                                   "sigma": np.diag(r * np.ones(self.ndims))}))
        self._self_normalize()
        self.model = CMixtureModel(models, weights)
