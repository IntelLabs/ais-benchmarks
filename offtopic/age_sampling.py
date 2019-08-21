import numpy as np
import matplotlib.pyplot as plt
from sampling_methods.base import t_tensor
from distributions.CGaussianMixtureModel import CGaussianMixtureModel


def displayPDF_1D(pdf, space_min, space_max):
    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot(111)
    plt.hold(True)
    plt.show(block=False)
    resolution = 0.02
    x = np.linspace(space_min, space_max, int((space_max - space_min) / resolution)).reshape(-1, 1)
    y = np.exp(pdf.log_prob(x))
    ax.plot(x, y, "-b", alpha=0.2)


space_min = t_tensor([0])
space_max = t_tensor([100])

means = t_tensor([[10], [42]])
covs = t_tensor([[80], [350]])
weights = t_tensor([0.2, 0.8])
gmm = CGaussianMixtureModel(means, covs, weights=weights)

displayPDF_1D(gmm, space_min, space_max)
plt.show(block=True)
