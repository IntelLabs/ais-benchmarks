import yaml
import numpy as np
import matplotlib.pyplot as plt
from ais_benchmarks.utils.misc import generateEggBoxGMM
from ais_benchmarks.utils.misc import generateRandomGMM
from ais_benchmarks.utils.plot_utils import plot_pdf2d
from ais_benchmarks.utils.plot_utils import plot_pdf


def make_normal(mu_min, mu_max, sigma_min, sigma_max):
    return make_gmm(mu_min, mu_max, sigma_min, sigma_max, n_modes=1)


def make_egg(mu_min, mu_max, sigma, n_modes):
    delta = (mu_max - mu_min) / n_modes
    return generateEggBoxGMM(mu_min, mu_max, delta, sigma)


def make_gmm(mu_min, mu_max, sigma_min, sigma_max, n_modes):
    support_min = mu_min - 5 * sigma_max
    support_max = mu_max + 5 * sigma_max
    return generateRandomGMM(support_min, support_max, n_modes, sigma_min, sigma_max)


def make_plot(filename, dist):
    # Plot only 1D and 2D dists
    fig = plt.figure()
    if d == 1:
        plot_pdf(plt.gca(), dist, space_min=dist.support_vals[0], space_max=dist.support_vals[1], resolution=0.01,
                 alpha=1)
    elif d == 2:
        plot_pdf2d(plt.gca(), dist, space_min=dist.support_vals[0], space_max=dist.support_vals[1], resolution=0.01,
                   alpha=1)
    else:
        return
    plt.savefig(filename, bbox_inches='tight', dpi=700)
    plt.close()


if __name__ == "__main__":
    import numpy as np

    name = "normal"
    for d in range(1, 7):
        dist = make_normal(mu_min=np.array([-5]*d), mu_max=np.array([5.]*d),
                           sigma_min=np.array([.01]*d), sigma_max=np.array([1]*d))
        with open("benchmarks/def_benchmark_%s%dD.yaml" % (name, dist.dims), "w+") as f:
            yaml.dump({"targets": [dist.to_dict(name=name, batch_size=2 ** dist.dims,
                                                nsamples=min(max(2000 * dist.dims, 10 ** dist.dims), 2e5),
                                                nsamples_eval=20000)]}, f)
        make_plot("benchmarks/def_benchmark_%s%dD.pdf" % (name, dist.dims), dist)

    name = "gmm"
    for d in range(1, 7):
        dist = make_gmm(mu_min=np.array([-5] * d), mu_max=np.array([5.] * d),
                        sigma_min=np.array([.01] * d), sigma_max=np.array([1] * d),
                        n_modes=5)
        with open("benchmarks/def_benchmark_%s%dD.yaml" % (name, dist.dims), "w+") as f:
            yaml.dump({"targets": [dist.to_dict(name=name, batch_size=2 ** dist.dims,
                                                nsamples=min(max(2000 * dist.dims, 10 ** dist.dims), 2e5),
                                                nsamples_eval=20000)]}, f)
        make_plot("benchmarks/def_benchmark_%s%dD.pdf" % (name, dist.dims), dist)

    name = "egg"
    for d in range(1, 7):
        dist = make_egg(mu_min=np.array([-5] * d), mu_max=np.array([5.] * d),
                        sigma=np.array([.1] * d), n_modes=5)
        with open("benchmarks/def_benchmark_%s%dD.yaml" % (name, dist.dims), "w+") as f:
            yaml.dump({"targets": [dist.to_dict(name=name, batch_size=2 ** dist.dims,
                                                nsamples=min(max(2000 * dist.dims, 10 ** dist.dims), 2e5),
                                                nsamples_eval=20000)]}, f)
        make_plot("benchmarks/def_benchmark_%s%dD.pdf" % (name, dist.dims), dist)

    # Make some figures with random distributions
    for d in range(1, 3):
        for i in range(0, 3):
            dist = make_normal(mu_min=np.array([-5]*d), mu_max=np.array([5.]*d),
                               sigma_min=np.array([.01]*d), sigma_max=np.array([1]*d))
            filename = "plots/dist_normal%dD_%d.pdf" % (dist.dims, i)
            make_plot(filename, dist)

            dist = make_gmm(mu_min=np.array([-5] * d), mu_max=np.array([5.] * d),
                            sigma_min=np.array([.01] * d), sigma_max=np.array([1] * d),
                            n_modes=5)
            filename = "plots/dist_gmm%dD_%d.pdf" % (dist.dims, i)
            make_plot(filename, dist)

        dist = make_egg(mu_min=np.array([-5] * d), mu_max=np.array([5.] * d),
                        sigma=np.array([.1] * d), n_modes=5)
        filename = "plots/dist_egg%dD.pdf" % dist.dims
        make_plot(filename, dist)

