import numpy as np
import matplotlib.pyplot as plt
from sampling_methods.base import t_tensor
from distributions.CGaussianMixtureModel import CGaussianMixtureModel
from distributions.CMultivariateNormal import CMultivariateNormal

from sampling_methods.tree_pyramid import CTreePyramidSampling


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
fig = plt.figure(figsize=(10, 7))
ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=1)
ax2 = plt.subplot2grid((3, 3), (0, 2), colspan=1, rowspan=1)
ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=1)
ax4 = plt.subplot2grid((3, 3), (1, 2), colspan=1, rowspan=1)
ax5 = plt.subplot2grid((3, 3), (2, 0), colspan=2, rowspan=1)
ax6 = plt.subplot2grid((3, 3), (2, 2), colspan=1, rowspan=1)
axes = [ax1, ax2, ax3, ax4, ax5, ax6]

space_min = t_tensor([-1])
space_max = t_tensor([1])
origin = (space_min + space_max) / 2.0
means = t_tensor([[-0.2], [-0.5], [0.70], [0.5], [0.7]])
covs = t_tensor([[0.005], [0.02], [0.01], [0.01],[0.005], ])
weights = t_tensor([0.2, 0.2, 0.2, 0.2, 0.2])
target_dist = CGaussianMixtureModel(means, covs, weights=weights)
displayPDF_1D(target_dist, space_min, space_max, ax=ax1)


# Tree pyramids
params = dict()
params["method"] = "simple"
params["resampling"] = "leaf"
params["kernel"] = "haar"
sampling_method = CTreePyramidSampling(space_min, space_max, params)
sampling_method.name = "sTP-AIS"

## TOP PLOT. WITH 4 SAMPLES
samples_acc, samples_w = sampling_method.importance_sample(target_d=target_dist,
                                                           n_samples=4,
                                                           timeout=90)
display_samples(samples_acc, np.zeros_like(samples_w), ax=ax1, marker="r|")
displayPDF_1D(target_dist, space_min, space_max, ax=ax1, color="g")
displayPDF_1D(sampling_method, space_min, space_max, ax=ax1, color="r")
display_samples(samples_acc, sampling_method.prob(samples_acc), ax=ax1, marker="rx")
sampling_method.T.plot(ax2)
ax2.axis("off")

## MIDDLE PLOT. WITH 6 SAMPLES
samples_acc, samples_w = sampling_method.importance_sample(target_d=target_dist,
                                                           n_samples=6,
                                                           timeout=90)
display_samples(samples_acc, np.zeros_like(samples_w), ax=ax3, marker="r|")
displayPDF_1D(target_dist, space_min, space_max, ax=ax3, color="g")
displayPDF_1D(sampling_method, space_min, space_max, ax=ax3, color="r")
display_samples(samples_acc, sampling_method.prob(samples_acc), ax=ax3, marker="rx")
sampling_method.T.plot(ax4)
ax4.axis("off")

## BOTTOM PLOT. WITH 8 SAMPLES
samples_acc, samples_w = sampling_method.importance_sample(target_d=target_dist,
                                                           n_samples=8,
                                                           timeout=90)
display_samples(samples_acc, np.zeros_like(samples_w), ax=ax5, marker="r|")
displayPDF_1D(target_dist, space_min, space_max, ax=ax5, color="g")
displayPDF_1D(sampling_method, space_min, space_max, ax=ax5, color="r")
display_samples(samples_acc, sampling_method.prob(samples_acc), ax=ax5, marker="rx")
sampling_method.T.plot(ax6)
ax6.axis("off")

ax1.set_ylabel("Density")
ax3.set_ylabel("Density")
ax5.set_ylabel("Density")
ax1.set_xlim(-1, 1)
plt.gcf().subplots_adjust(bottom=0.2)
plt.tight_layout()
plt.show(block=True)
