import numpy as np
from numpy import array as t_tensor
import itertools
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from sampling_experiments.sampling_methods.base import CSamplingMethod

class CTreePyramidNode:
    def __init__(self, center, idx, radius, level, nidx):
        self.center = center
        self.radius = radius
        self.children = [None] * (2**len(center))
        self.value = t_tensor([0])
        self.leaf_idx = idx
        self.node_idx = nidx
        self.coords = t_tensor(center)
        self.level = level

    def __lt__(self, other):
        return self.value < other.value

    def is_leaf(self):
        return self.leaf_idx != -1

    def __float__(self):
        return self.value

    def __repr__(self):
        return "[ " + str(self.leaf_idx) +" , "+ str(self.radius) +  " ]"

class CTreePyramid:
    def __init__(self, dmin, dmax):
        self.root = CTreePyramidNode(center=(dmin + dmax) / 2, idx=0, radius=np.max((dmax - dmin) / 2), level=0, nidx=0)
        self.nodes = [self.root]
        self.leaves = [self.root]
        self.axis = None
        self.ndims = len(dmin)

    def expand(self, particle):

        # Mark the particle to expand
        if self.axis is not None and self.ndims==1:
            x = particle.center - particle.radius
            rect = patches.Rectangle([x, 0], particle.radius*2, particle.value / (particle.radius*2), fill=False, linewidth=3, linestyle="--", alpha=1, color="g", zorder=10)
            self.axis.add_patch(rect)

        center_coeffs = itertools.product([-1,1], repeat=len(particle.center))

        p_centers = []
        for coeff in center_coeffs:
            offset = (particle.radius/2) * t_tensor(coeff)
            p_centers.append(particle.center + offset)

        # The first new leaf particle replaces the expanding leaf particle in the leaves list and indices
        p1 = CTreePyramidNode(center=p_centers[0], idx=particle.leaf_idx, radius=particle.radius/2, level=particle.level+1, nidx=len(self.nodes))
        self.leaves[particle.leaf_idx] = p1
        self.nodes.append(p1)
        particle.children = [p1]
        particle.leaf_idx = -1

        # The rest of the particles are new and appended to the tree lists
        new_particles = []
        for i,center in enumerate(p_centers[1:]):
            new_particles.append(CTreePyramidNode(center=center, idx=len(self.leaves) + i, radius=particle.radius/2, level=particle.level+1, nidx=len(self.nodes)))

        for new_p in new_particles:
            self.leaves.append(new_p)
            self.nodes.append(new_p)
            particle.children.append(new_p)

        return particle.children

    @staticmethod
    def draw_node(axis, x, y, label="x", color="b"):
        axis.add_patch(patches.Circle((x,y), 0.1, facecolor="w", linewidth=1, edgecolor=color))
        axis.annotate(label, xy=(x, y), fontsize=12, ha="center", va="center")
        axis.plot(x,y)

    @staticmethod
    def plot_node(axis, node, x, y):
        plot_span = 10
        if node.is_leaf():
            CTreePyramid.draw_node(axis,x,y, "$x_{%d}$" % node.node_idx, color="r")
        else:
            CTreePyramid.draw_node(axis, x, y, "$x_{%d}$" % node.node_idx, color="b")
            for idx,ch in enumerate(node.children):
                nodes_in_level = 2 ** (len(node.coords) * ch.level)
                if nodes_in_level>1:
                    increment = plot_span / nodes_in_level
                else:
                    increment = 0
                x_ch = x + increment * idx - (plot_span/nodes_in_level) / 2
                y_ch = -ch.level
                axis.arrow(x, y, x_ch - x, y_ch - y, alpha=0.2, zorder=0)
                CTreePyramid.plot_node(axis, ch, x_ch, y_ch)

    def plot(self, axis):
        axis.cla()
        self.plot_node(axis, self.root, 0, 0)


class CTreePyramidSampling(CSamplingMethod):
    def __init__(self, space_min, space_max):
        super(self.__class__, self).__init__(space_min, space_max)
        self.range = space_max - space_min
        self.T = CTreePyramid(space_min, space_max)
        self.particles_to_expand = [self.T.root]
        self.ndims = len(space_min)
        self.integral = 0

    def reset(self):
        self.T = CTreePyramid(self.space_min, self.space_max)
        self.particles_to_expand = [self.T.root]

    def expand_particles(self, particles, max_parts):
        new_particles = []
        remaining_particles = []
        particles_coords = t_tensor([])
        processed_particles = 0
        for p in particles:
            new_parts = self.T.expand(p)
            for npart in new_parts:
                particles_coords = np.concatenate((particles_coords, npart.coords))
            new_particles.extend(new_parts)
            processed_particles = processed_particles + 1
            if len(new_particles) > max_parts:
                break

        if len(new_particles) > max_parts:
            remaining_particles = particles[processed_particles:]

        return new_particles, particles_coords, remaining_particles

    def samplTPyramidPDF(self, pdf, n_samples=10000):
        # Expand particles and stop after n_samples have been taken
        values_acc = t_tensor([])
        samples_acc = t_tensor([])
        num_evals = 0
        while len(self.particles_to_expand) > 0 and n_samples > len(values_acc):
            self.particles_to_expand.sort(reverse=True)
            eval_parts, samples, self.particles_to_expand = self.expand_particles(self.particles_to_expand, n_samples)
            num_evals = num_evals + len(eval_parts)

            values = np.exp(pdf.log_prob(samples.reshape(-1, self.ndims)))
            samples_acc = np.concatenate((samples_acc, samples))
            values_acc = np.concatenate((values_acc, values))

            # Compute the density for each particle and add it to the expansion set
            for idx_int in range(len(eval_parts)):
                eval_parts[idx_int].value = values[idx_int] * ((eval_parts[idx_int].radius*2) ** self.ndims)
                self.particles_to_expand.append(eval_parts[idx_int])

        volume = 0
        for n in self.T.leaves:
            volume = volume + n.value
        self.integral = volume

        return samples_acc.reshape((-1, self.ndims)), np.log(values_acc)

    def sample(self, n_samples):
        raise NotImplementedError

    def sample_with_likelihood(self, pdf, n_samples):
        samples, values = self.samplTPyramidPDF(pdf,n_samples)
        return samples, values

