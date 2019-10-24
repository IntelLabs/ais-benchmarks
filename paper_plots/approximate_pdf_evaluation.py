import numpy as np
import matplotlib.pyplot as plt
from sampling_methods.base import t_tensor
from distributions.CGaussianMixtureModel import CGaussianMixtureModel
from sampling_methods.tree_pyramid import CTreePyramidSampling
from sampling_methods.m_pmc import CMixturePMC
from distributions.CKernelDensity import CKernelDensity
from distributions.CNearestNeighbor import CNearestNeighbor
from sampling_methods.evaluation import kl_divergence_components
from sampling_methods.evaluation import js_divergence


def displayPDF_1D(pdf, space_min, space_max, ax=None, alpha=1.0, color=None):
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = plt.subplot(111)
    plt.hold(True)
    plt.show(block=False)
    resolution = 0.02
    x = np.linspace(space_min, space_max, int((space_max - space_min) / resolution)).reshape(-1, 1)
    y = pdf.prob(x)
    ax.plot(x, y, "-", alpha=alpha, color=color)


def display_samples(samples, samples_prob, ax=None, marker="x", alpha=1.0):
    ax.plot(samples, samples_prob, marker, alpha=alpha)


# Figure Sampling algorithm evaluation sequence.
# Left: Ground truth PDF.
fig = plt.figure(figsize=(10, 3.5))
ax1 = plt.subplot(131)
ax2 = plt.subplot(132, sharey=ax1, sharex=ax1)
ax3 = plt.subplot(133, sharey=ax1, sharex=ax1)

space_min = t_tensor([0])
space_max = t_tensor([1])
means = t_tensor([[0.1], [0.5], [0.7]])
covs = t_tensor([[0.01], [0.02], [0.01]])
weights = t_tensor([0.3, 0.5, 0.2])
target_dist = CGaussianMixtureModel(means, covs, weights=weights)
displayPDF_1D(target_dist, space_min, space_max, ax=ax1)


# Middle: Samples generated from the evaluated sampling algorithm and approximated PDF.
max_samples = 50

# Tree pyramids
# params = dict()
# params["method"] = "simple"
# params["resampling"] = "full"
# params["kernel"] = "haar"
# sampling_method = CTreePyramidSampling(space_min, space_max, params)
# sampling_method.name = "sTP-AIS"

# M-PMC
params = dict()
params["K"] = 20  # Number of samples per proposal distribution
params["N"] = 10  # Number of proposal distributions
params["J"] = 1000
params["sigma"] = 0.001  # Scaling parameter of the proposal distributions
sampling_method = CMixturePMC(space_min, space_max, params)
sampling_method.name = "M-PMC"

samples_acc, samples_w = sampling_method.importance_sample(target_d=target_dist,
                                                           n_samples=max_samples,
                                                           timeout=90)

approximate_pdf = sampling_method
# approximate_pdf = CKernelDensity(samples_acc, samples_w, bw=0.06)

display_samples(samples_acc, np.zeros_like(samples_w), ax=ax2, marker="r|")
displayPDF_1D(approximate_pdf, space_min, space_max, ax=ax2, color="g")
display_samples(samples_acc, approximate_pdf.prob(samples_acc), ax=ax2, marker="rx")


# Right: Distance measurement between ground truth and approximated distribution
displayPDF_1D(target_dist, space_min, space_max, ax=ax3, alpha=0.4)
displayPDF_1D(approximate_pdf, space_min, space_max, ax=ax3, alpha=0.4)

resolution = 0.01
samples = np.linspace(space_min, space_max, int((space_max - space_min) / resolution)).reshape(-1, 1)
ax4 = ax3.twinx()
ax4.spines["right"].set_edgecolor("r")
ax4.tick_params(axis='y', colors="r")
ax4.set_ylabel("$JSD(\\pi||Q)$")
ax4.yaxis.label.set_color("r")
display_samples(samples, js_divergence(target_dist.prob(samples),
                                       approximate_pdf.prob(samples)),
                ax=ax4, marker="r-")

ax1.set_ylabel("Density")
ax1.set_xlabel("(a)")
ax2.set_xlabel("(b) N=%d N-ESS=%.3f" % (max_samples, sampling_method.get_NESS()))
ax3.set_xlabel("(c)")
ax3.set_xlim(0,1)
plt.gcf().subplots_adjust(bottom=0.2)
plt.tight_layout()
plt.show(block=True)
