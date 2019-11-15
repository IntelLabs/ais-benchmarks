import numpy as np
import matplotlib.pyplot as plt
from sampling_methods.base import t_tensor
from distributions.CGaussianMixtureModel import CGaussianMixtureModel
from sampling_methods.base import grid_sample_distribution
from utils.misc import generateRandomGMM
from utils.misc import generateEggBoxGMM


def plot_grid_sampled_pdfs(ax, dims, prob_p, shape=None, marginalize_axes = None, alpha=1):
    prob_p_margin = prob_p.reshape(shape).transpose()
    if marginalize_axes is not None:
        prob_p_margin = np.sum(prob_p_margin, axis=marginalize_axes)

    X = np.array(dims[0]).reshape(-1)
    Y = np.array(dims[1]).reshape(-1)
    ax.contourf(X, Y, prob_p_margin.reshape(len(dims[0]), len(dims[1])), cmap=plt.cm.Reds)


def displayPDF_2D(ax, pdf, space_min, space_max):
    grid, log_prob, dims, shape = grid_sample_distribution(pdf, space_min, space_max, resolution=0.02)
    plot_grid_sampled_pdfs(ax, dims, np.exp(log_prob), shape=None, marginalize_axes=None, alpha=1)
    plt.show(block=False)
    plt.pause(0.01)


# Figure Target distributions.
space_min = t_tensor([-1, -1])
space_max = t_tensor([1, 1])

# Left: 2D Normal ground truth PDF.
fig = plt.figure(figsize=(12, 3.5))
ax1 = plt.subplot(131)
ax2 = plt.subplot(132, sharey=ax1)
ax3 = plt.subplot(133, sharey=ax1)

means = t_tensor([[0, 0]])
covs = t_tensor([[0.1, 0.1]])
weights = t_tensor([1])
target_dist = CGaussianMixtureModel(means, covs, weights=weights)
displayPDF_2D(ax1,target_dist,space_min, space_max)

# Middle: 2D 5 Components GMM
target_dist = generateRandomGMM(space_min+0.2, space_max-0.2, 5, sigma_min=[0.01, 0.01], sigma_max=[0.05, 0.05])
displayPDF_2D(ax2, target_dist, space_min, space_max)
plt.hold(True)
for i, loc in enumerate(target_dist.mu):
    ax2.scatter(loc[1], loc[0], marker="x")
    if loc[1] > 0:
        loc[1] -= 0.28
    else:
        loc[1] += 0.05
    ax2.text(loc[1], loc[0], "$\mu_%d$" % i, fontsize=18)
plt.hold(False)


# Right: 2D Egg carton
target_dist = generateEggBoxGMM(space_min+0.2, space_max-0.2, 0.4, sigma=[0.01, 0.01])
displayPDF_2D(ax3, target_dist, space_min, space_max)

ax2.set_xlim(-1,1)
ax1.set_xlabel("(a) Normal with \n $\mu = [%1.2f, %1.2f]$ and $\sigma=[%1.2f, %1.2f]$" % (means[0, 0], means[0, 1], covs[0, 0], covs[0, 1],))
ax2.set_xlabel("(b) GMM with 5 mixture \ncomponents. ")
ax3.set_xlabel("(c) Egg carton distribution with 4 modes \n per dimension and $\Sigma=diag(0.01,0.01)$")
plt.gcf().subplots_adjust(bottom=0.2)
plt.tight_layout()
plt.show(block=True)
