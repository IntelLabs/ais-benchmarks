import numpy as np
import matplotlib.pyplot as plt
import time
import os

from distributions.CGaussianMixtureModel import CGaussianMixtureModel
from distributions.CMultivariateUniform import CMultivariateUniform
from distributions.CMultivariateNormal import CMultivariateNormal
from distributions.CKernelDensity import CKernelDensity
from sampling_methods.base import t_tensor


def f(x):
    return np.sin(5*x)



space_min = t_tensor([-1])
space_max = t_tensor([1])
means = t_tensor([[-0.1], [0.5], [0.7]])
covs = t_tensor([[0.01], [0.02], [0.01]])
weights = t_tensor([0.3, 0.5, 0.2])
target_dist = CGaussianMixtureModel(means, covs, weights=weights)
prior_dist = CMultivariateUniform((space_min+space_max)/2, (space_max-space_min)/2)

resolution = 0.001
x = np.linspace(space_min, space_max, int((space_max - space_min) / resolution)).reshape(-1, 1)

xpix = f(x).flatten()*target_dist.prob(x).flatten()
xsimtrue = target_dist.sample(10000)
xsimpixtrue = f(xsimtrue).flatten()*target_dist.prob(xsimtrue).flatten()


# # # First plot: Prior
# fig = plt.figure(figsize=(10, 3.5))
# ax1 = plt.subplot(111)
# ax1.plot(x, prior_dist.prob(x), "-", alpha=1, color="k", label="$\pi(x)$")
# ax1.legend()
# ax1.set_ylim(0, 3)
#
# First plot 2: Likelihood
# fig = plt.figure(figsize=(10, 3.5))
# ax1 = plt.subplot(111)
# ax1.plot(x, target_dist.prob(x)*2, "-", alpha=1, color="b", label="$\pi(D|x)$")
# ax1.fill_between(x.flatten(), target_dist.prob(x)*2, color="g", alpha=0.2, label="$\int\pi(D|x)dx$")
# ax1.set_ylim(0, 3)
# ax1.legend()
# plt.show(block=False)
#
# # Second plot: Posterior
# fig = plt.figure(figsize=(10, 3.5))
# ax1 = plt.subplot(111)
# ax1.plot(x, target_dist.prob(x), "-", alpha=1, color="r", label="$\pi(x|D)$")
# plt.fill_between(x[x<0].flatten(), target_dist.prob(x[x<0].reshape(-1,1)), color="r", alpha=0.2, label="$\int_{-1}^0\pi(x|D)dx$")
# ax1.set_ylim(0, 3)
# ax1.legend()
# plt.show(block=False)

# # Third plot: Posterior * Function
# fig = plt.figure(figsize=(10, 3.5))
# ax1 = plt.subplot(111)
# ax1.plot(x, f(x), "-", alpha=1, color="b", label="$f(x)$")
# ax1.plot(x, xpix, "-", alpha=1, color="g", label="$f(x)\pi(x)$")
# plt.fill_between(x.flatten(), xpix, color="g", alpha=0.2)
# ax1.legend()

# # Fourth plot: Monte Carlo approximation of a probability. Reject approach
# fig = plt.figure(figsize=(10, 3.5))
# ax1 = plt.subplot(111)
# xsim = target_dist.sample(1)
# true_prob = np.sum(xsimtrue < 0) / len(xsimtrue)
# for i in range(1, 200, 2):
#     xsim = np.concatenate((xsim, target_dist.sample(2)))
#     ax1.plot(x, target_dist.prob(x), "-", alpha=1, color="r", label="$\pi(x|D)$")
#     # ax1.plot(xsim, np.zeros_like(xsim), ".", alpha=1, color="r", label="$x_i\sim\pi(x|D)$")
#     ax1.plot(xsim[xsim < 0], np.zeros_like(xsim[xsim < 0]), "o", alpha=1, color="g", label="$x_i\sim\pi(x|D, x < 0) $")
#     ax1.plot(xsim[xsim >= 0], np.zeros_like(xsim[xsim >= 0]), "x", alpha=1, color="r", label="$x_i\sim\pi(x|D, x \geq 0) $")
#     prob = np.sum(xsim < 0) / len(xsim)
#     ax1.set_title("$\pi(x|D, x < 0) \\approx %3.5f.$ Ground truth: $%3.5f$. N=%d" % (prob, true_prob, len(xsim)))
#
#     ax1.legend()
#     plt.savefig("mcapprox" + os.sep + "mcapprox_%d.png" % i)
#     fig.canvas.draw()
#     fig.canvas.flush_events()
#     plt.show(block=False)
#     time.sleep(0.1)
#     ax1.cla()

# # Fourth plot: Monte Carlo approximation of a probability. Mean approach
# fig = plt.figure(figsize=(10, 3.5))
# ax1 = plt.subplot(111)
# xsim = target_dist.sample(1)
# true_prob = np.sum(xsimtrue < 0) / len(xsimtrue)
# mean_dist = CMultivariateUniform(t_tensor([-0.5]), t_tensor([0.5]))
# xsim = mean_dist.sample(1)
#
# for i in range(1, 200, 2):
#     xsim = np.concatenate((xsim, mean_dist.sample(2)))
#     ax1.plot(x, target_dist.prob(x), "-", alpha=1, color="r", label="$\pi(x|D)$")
#     # ax1.plot(xsim, np.zeros_like(xsim), ".", alpha=1, color="r", label="$x_i\sim\pi(x|D)$")
#     ax1.plot(xsim, np.zeros_like(xsim), ".", alpha=1, color="g", label="$x_i \sim U(-1,0) $")
#     prob = np.mean(target_dist.prob(xsim))
#     ax1.set_title("$\pi(x|D, x < 0) \\approx %3.5f.$ Ground truth: $%3.5f$. N=%d" % (prob, true_prob, len(xsim)))
#
#     ax1.legend()
#     plt.savefig("mcapprox" + os.sep + "mcapprox_%d.png" % i)
#     fig.canvas.draw()
#     fig.canvas.flush_events()
#     plt.show(block=False)
#     time.sleep(0.1)
#     ax1.cla()

#
# # Fourth plot: Monte Carlo approximation with ratio of inliers.
# fig = plt.figure(figsize=(10, 3.5))
# ax1 = plt.subplot(111)
# xsim = target_dist.sample(1)
# for i in range(1, 200, 2):
#     xsim = np.concatenate((xsim, target_dist.sample(2)))
#     xsimpix = f(xsim).flatten()*target_dist.prob(xsim).flatten()
#     ax1.plot(x, target_dist.prob(x), "-", alpha=1, color="r", label="$\pi(x|D)$")
#     ax1.plot(xsim, np.zeros_like(xsim), ".", alpha=1, color="r", label="$x_i\sim\pi(x|D)$")
#     ax1.plot(x, xpix, "-", alpha=1, color="g", label="$f(x)\pi(x)$")
#     ax1.plot(np.mean(xsimpixtrue), 0, "*", alpha=1, color="g", label="$E[f(x)]$", markersize=12, markeredgecolor="k")
#     ax1.plot(np.mean(xsimpix), 0, "*", alpha=1, color="r", label="$\hat{E}[f(x)]$", markersize=12, markeredgecolor="k")
#     plt.vlines(xsim[xsimpix < 0], ymin=xsimpix[xsimpix < 0], ymax=np.zeros_like(xsimpix[xsimpix < 0]), linewidth=1, linestyles="dashed", colors="g")
#     plt.vlines(xsim[xsimpix > 0], ymin=np.zeros_like(xsimpix[xsimpix > 0]), ymax=xsimpix[xsimpix > 0], linewidth=1, linestyles="dashed", colors="g")
#     ax1.legend()
#     plt.savefig("mcapprox" + os.sep + "mcapprox_%d.png" % i)
#     fig.canvas.draw()
#     fig.canvas.flush_events()
#     plt.show(block=False)
#     time.sleep(0.1)
#     ax1.cla()

# # Fifth plot: Importance sampling
# fig = plt.figure(figsize=(10, 3.5))
# ax1 = plt.subplot(111)
# true_prob = np.sum(xsimtrue < 0) / len(xsimtrue)
# proposal_dist = CMultivariateNormal(t_tensor([-0.2]), t_tensor([[0.02]]))
# xsim = proposal_dist.sample(1)
#
# for i in range(1, 200, 2):
#     ax1.cla()
#     xsim = np.concatenate((xsim, proposal_dist.sample(2)))
#     xsim = xsim[xsim < 0].reshape(-1, 1)
#     ax1.plot(x, target_dist.prob(x), "-", alpha=1, color="r", label="$\pi(x|D)$")
#     ax1.plot(x, proposal_dist.prob(x), "-", alpha=1, color="g", label="$Q(x)$")
#     weights = target_dist.prob(xsim) / proposal_dist.prob(xsim)
#     plt.vlines(xsim, ymin=np.zeros_like(weights), ymax=weights * proposal_dist.prob(xsim), linewidth=1, linestyles="dashed", colors="g", label="$w_i = \pi(x|D)/Q(x)$")
#
#     # ax1.plot(xsim, np.zeros_like(xsim), ".", alpha=1, color="r", label="$x_i\sim\pi(x|D)$")
#     ax1.plot(xsim, np.zeros_like(xsim), ".", alpha=1, color="g", label="$x_i \sim Q $")
#     prob = weights * target_dist.prob(xsim)
#     ax1.set_title("$\pi(x|D, x < 0) \\approx %3.5f.$ Ground truth: $%3.5f$. N=%d" % (np.mean(prob), true_prob, len(xsim)))
#
#     ax1.legend()
#     ax1.set_ylim(0, 3)
#     plt.savefig("importance_sampling" + os.sep + "IS_%d.png" % i)
#     fig.canvas.draw()
#     fig.canvas.flush_events()
#     plt.show(block=False)
#     time.sleep(0.1)

#
# # Sixth plot: Importance Rejection sampling
# fig = plt.figure(figsize=(10, 3.5))
# ax1 = plt.subplot(111)
# proposal_dist = CMultivariateNormal(t_tensor([0.1]), t_tensor([[0.2]]))
# xsim = proposal_dist.sample(1)
#
# kde = CKernelDensity(xsim, np.ones(len(xsim)) / len(xsim))
# kde.bw = 0.01
#
# for i in range(1, 200, 1):
#     ax1.cla()
#     xsim = np.concatenate((xsim, proposal_dist.sample(1)))
#
#     ax1.plot(x, target_dist.prob(x), "-", alpha=1, color="r", label="$\pi(x|D)$")
#     ax1.plot(x, proposal_dist.prob(x), "-", alpha=1, color="g", label="$Q(x)$")
#
#     plt.vlines(xsim[-1], ymin=0, ymax=proposal_dist.prob(xsim[-1]), linewidth=1, linestyles="dashed", colors="g", label="$Q(x_i)$")
#     y_sim = np.random.uniform(0, proposal_dist.prob(xsim[-1]))
#     ax1.plot(xsim[-1], y_sim, "x", alpha=1, color="g", label="$y \sim U(0, Q(x_i))$")
#     if y_sim > target_dist.prob(xsim[-1]):
#         xsim = xsim[0:-1]  # Reject
#
#     weights = target_dist.prob(xsim) / proposal_dist.prob(xsim)
#     kde.samples = xsim
#     kde.weights = weights
#     kde.fit()
#     ax1.plot(x, kde.prob(x), "--", alpha=1, color="r", label="$\hat{\pi}(x)$")
#     ax1.plot(xsim, np.zeros_like(xsim), ".", alpha=1, color="r", label="$x_i \sim \pi(x|D) $")
#
#     ax1.legend()
#     ax1.set_ylim(0, 3)
#     plt.savefig("rejection" + os.sep + "reject_IS_%d.png" % i)
#     fig.canvas.draw()
#     fig.canvas.flush_events()
#     plt.show(block=False)
#     time.sleep(0.01)

# Seventh plot: Multiple Importance sampling
fig = plt.figure(figsize=(10, 3.5))
ax1 = plt.subplot(111)
proposal_dists = [CMultivariateNormal(t_tensor([-0.1]), t_tensor([[0.02]])),
                  CMultivariateNormal(t_tensor([0.5]), t_tensor([[0.02]]))]
# Initial sample
proposal_dist = proposal_dists[np.random.randint(0, len(proposal_dists))]
xsim = proposal_dist.sample(1)
weights = target_dist.prob(xsim[-1]) / proposal_dist.prob(xsim[-1])

kde = CKernelDensity(xsim, np.ones(len(xsim)) / len(xsim))
kde.bw = 0.01

for i in range(1, 200, 1):
    ax1.cla()
    proposal_dist = proposal_dists[np.random.randint(0, len(proposal_dists))]

    xsim = np.concatenate((xsim, proposal_dist.sample(1)))

    ax1.plot(x, target_dist.prob(x), "-", alpha=1, color="r", label="$\pi(x|D)$")

    for j, Q in enumerate(proposal_dists):
        ax1.plot(x, Q.prob(x), "--", alpha=1, color="g", label="$Q_%d(x)$" % j)

    plt.vlines(xsim[-1], ymin=0, ymax=proposal_dist.prob(xsim[-1]), linewidth=1, linestyles="dashed", colors="g")

    weights = np.concatenate((weights, target_dist.prob(xsim[-1]) / proposal_dist.prob(xsim[-1])))

    kde.samples = xsim
    kde.weights = weights
    kde.fit()
    ax1.plot(x, kde.prob(x), "--", alpha=1, color="r", label="$\hat{\pi}(x)$")
    ax1.plot(xsim, np.zeros_like(xsim), ".", alpha=1, color="r", label="$x_i \sim \pi(x|D) $")

    ax1.legend()
    ax1.set_ylim(0, 3)
    plt.savefig("multipleIS" + os.sep + "multiple_IS_%d.png" % i)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.show(block=False)
    time.sleep(0.01)

# Eight plot: Adaptive Multiple Importance sampling


plt.show(block=True)
