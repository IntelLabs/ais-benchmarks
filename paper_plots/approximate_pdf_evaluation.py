import numpy as np
import matplotlib.pyplot as plt
from sampling_methods.base import t_tensor
from distributions.CGaussianMixtureModel import CGaussianMixtureModel
from sampling_methods.tree_pyramid import CTreePyramidSampling
from distributions.CKernelDensity import CKernelDensity
from distributions.CNearestNeighbor import CNearestNeighbor
from sampling_methods.evaluation import kl_divergence_components


def displayPDF_1D(pdf, space_min, space_max, ax=None, alpha=1.0, color=None):
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = plt.subplot(111)
    plt.hold(True)
    plt.show(block=False)
    resolution = 0.02
    x = np.linspace(space_min, space_max, int((space_max - space_min) / resolution)).reshape(-1, 1)
    y = np.exp(pdf.log_prob(x))
    ax.plot(x, y, "-", alpha=alpha, color=color)


def display_samples(samples, samples_prob, ax=None, marker="x", alpha=1.0):
    ax.plot(samples, samples_prob, marker, alpha=alpha)


# Figure Sampling algorithm evaluation sequence.
# Left: Ground truth PDF.
fig = plt.figure(figsize=(10, 3.5))
ax1 = plt.subplot(131)
ax2 = plt.subplot(132, sharey=ax1)
ax3 = plt.subplot(133, sharey=ax1)

space_min = t_tensor([0])
space_max = t_tensor([1])
means = t_tensor([[0.1], [0.5], [0.7]])
covs = t_tensor([[0.01], [0.02], [0.01]])
weights = t_tensor([0.3, 0.5, 0.2])
target_dist = CGaussianMixtureModel(means, covs, weights=weights)
displayPDF_1D(target_dist, space_min, space_max, ax=ax1)


# Middle: Samples generated from TP sampling algorithm and approximated PDF.
tp_sampling_method = CTreePyramidSampling(space_min, space_max)

max_samples = 20
samples_acc = t_tensor([])
samples_logprob_acc = t_tensor([])
while len(samples_acc) < max_samples:
    samples, samples_logprob = tp_sampling_method .sample_with_likelihood(pdf=target_dist, n_samples=2)
    if len(samples_acc) > 0:
        samples_acc = np.vstack((samples_acc, samples))
    else:
        samples_acc = samples
    samples_logprob_acc = np.concatenate((samples_logprob_acc, samples_logprob))

approximate_pdf = CKernelDensity(samples_acc, np.exp(samples_logprob_acc), bw=0.05)
# approximate_pdf = CNearestNeighbor(samples_acc, samples_logprob_acc)
display_samples(samples_acc, np.zeros_like(samples_logprob_acc), ax=ax2, marker="r|")
displayPDF_1D(approximate_pdf, space_min, space_max, ax=ax2, color="g")
display_samples(samples_acc, np.exp(approximate_pdf.log_prob(samples_acc)), ax=ax2, marker="rx")


# Right: Distance measurement between ground truth and approximated distribution
displayPDF_1D(target_dist, space_min, space_max, ax=ax3, alpha=0.4)
displayPDF_1D(approximate_pdf, space_min, space_max, ax=ax3, alpha=0.4)

resolution = 0.01
samples = np.linspace(space_min, space_max, int((space_max - space_min) / resolution)).reshape(-1, 1)
ax4 = ax3.twinx()
ax4.spines["right"].set_edgecolor("r")
ax4.tick_params(axis='y', colors="r")
ax4.set_ylabel("KL Divergence")
ax4.yaxis.label.set_color("r")
display_samples(samples, kl_divergence_components(np.exp(target_dist.log_prob(samples)),
                                                  np.exp(approximate_pdf.log_prob(samples))),
                ax=ax4, marker="r-")

ax1.set_ylabel("Density")
ax1.set_xlabel("(a) Ground truth \ndistribution p(x)")
ax2.set_xlabel("(b) Samples and approximated\n distribution: q(x)")
ax3.set_xlabel("(c) KL(p||q)")
plt.gcf().subplots_adjust(bottom=0.2)
plt.tight_layout()
plt.show(block=True)
