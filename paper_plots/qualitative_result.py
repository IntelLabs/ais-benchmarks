import numpy as np
import matplotlib.pyplot as plt
from sampling_methods.base import t_tensor
from distributions.CGaussianMixtureModel import CGaussianMixtureModel
from distributions.CMultivariateNormal import CMultivariateNormal

from sampling_methods.tree_pyramid import CTreePyramidSampling
from sampling_methods.m_pmc import CMixturePMC
from sampling_methods.layered_ais import CLayeredAIS
from sampling_methods.dm_ais import CDeterministicMixtureAIS
from sampling_methods.multi_nested import CMultiNestedSampling
from sampling_methods.metropolis_hastings import CMetropolisHastings

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
fig = plt.figure(figsize=(10, 7))
ax1 = plt.subplot(231)
ax2 = plt.subplot(232, sharey=ax1, sharex=ax1)
ax3 = plt.subplot(233, sharey=ax1, sharex=ax1)
ax4 = plt.subplot(234, sharey=ax1, sharex=ax1)
ax5 = plt.subplot(235, sharey=ax1, sharex=ax1)
ax6 = plt.subplot(236, sharey=ax1, sharex=ax1)
axes = [ax1, ax2, ax3, ax4, ax5, ax6]

space_min = t_tensor([-1])
space_max = t_tensor([1])
origin = (space_min + space_max) / 2.0
means = t_tensor([[-0.2], [-0.5], [0.70], [0.5], [0.7]])
covs = t_tensor([[0.005], [0.02], [0.01], [0.01],[0.005], ])
weights = t_tensor([0.2, 0.2, 0.2, 0.2, 0.2])
target_dist = CGaussianMixtureModel(means, covs, weights=weights)
displayPDF_1D(target_dist, space_min, space_max, ax=ax1)


# Middle: Samples generated from the evaluated sampling algorithm and approximated PDF.
max_samples = 50

sampling_methods = []

# Tree pyramids
params = dict()
params["method"] = "simple"
params["resampling"] = "leaf"
params["kernel"] = "haar"
sampling_method = CTreePyramidSampling(space_min, space_max, params)
sampling_method.name = "sTP-AIS"
sampling_methods.append(sampling_method)

# M-PMC
params = dict()
params["K"] = 20  # Number of samples per proposal distribution
params["N"] = 10  # Number of proposal distributions
params["J"] = 1000
params["sigma"] = 0.001  # Scaling parameter of the proposal distributions
sampling_method = CMixturePMC(space_min, space_max, params)
sampling_method.name = "M-PMC[2]"
sampling_methods.append(sampling_method)

# Layered Deterministic Mixture Adaptive Importance Sampling
params["K"] = 3  # Number of samples per proposal distribution
params["N"] = 5  # Number of proposal distributions
params["J"] = 1000  # Total number of samples
params["L"] = 10  # Number of MCMC moves during the proposal adaptation
params["sigma"] = 0.01  # Scaling parameter of the proposal distributions
params["mh_sigma"] = 0.005  # Scaling parameter of the mcmc proposal distributions moment update
tp_sampling_method = CLayeredAIS(space_min, space_max, params)
tp_sampling_method.name = "LAIS[3]"
sampling_methods.append(tp_sampling_method)

# Deterministic Mixture Adaptive Importance Sampling
params["K"] = 5  # Number of samples per proposal distribution
params["N"] = 10  # Number of proposal distributions
params["J"] = 1000
params["sigma"] = 0.01  # Scaling parameter of the proposal distributions
tp_sampling_method = CDeterministicMixtureAIS(space_min, space_max, params)
tp_sampling_method.name = "DM-AIS[1]"
sampling_methods.append(tp_sampling_method)

# Multi-Nested sampling
MCMC_proposal_dist = CMultivariateNormal(origin, np.diag(np.ones_like(space_max)) * 0.01)
params["proposal"] = MCMC_proposal_dist
params["N"] = 30
params["kde_bw"] = 0.01  # Bandwidth of the KDE approximation to evaluate the prob of the distribution approximated by the set of generated samples
mnested_sampling_method = CMultiNestedSampling(space_min, space_max, params)
mnested_sampling_method.name = "Multi-Nested[27]"
sampling_methods.append(mnested_sampling_method)

# Metropolis-Hastings
MCMC_proposal_dist = CMultivariateNormal(origin, np.diag(np.ones_like(space_max)) * 0.1)
params["proposal_d"] = MCMC_proposal_dist  # MC move proposal distribution p(x'|x)
params["n_steps"] = 2  # Num of decorrelation steps: discarded samples upon new accept
params["n_burnin"] = 10  # Number of samples considered as burn-in
params[
    "kde_bw"] = 0.01  # Bandwidth of the KDE approximation to evaluate the prob of the distribution approximated by the set of generated samples
mh_sampling_method = CMetropolisHastings(space_min, space_max, params)
mh_sampling_method.name = "MCMC-MH[8]"
sampling_methods.append(mh_sampling_method)

letters = "abcdef"
for sampling_method, ax, letter in zip(sampling_methods, axes, letters):
    samples_acc, samples_w = sampling_method.importance_sample(target_d=target_dist,
                                                               n_samples=max_samples,
                                                               timeout=90)
    approximate_pdf = sampling_method

    display_samples(samples_acc, np.zeros_like(samples_w), ax=ax, marker="r|")
    displayPDF_1D(target_dist, space_min, space_max, ax=ax, color="g")
    displayPDF_1D(approximate_pdf, space_min, space_max, ax=ax, color="r")
    display_samples(samples_acc, approximate_pdf.prob(samples_acc), ax=ax, marker="rx")
    ax.set_xlabel("(%s) %s" % (letter, sampling_method.name))

ax1.set_ylabel("Density")
ax4.set_ylabel("Density")
ax1.set_xlim(-1, 1)
plt.gcf().subplots_adjust(bottom=0.2)
plt.tight_layout()
plt.show(block=True)
