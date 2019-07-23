import time
import numpy as np
from numpy import array as t_tensor
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import sys

# PROBABILITY DISTRIBUTION FUNCTIONS
from distributions.CGaussianMixtureModel import CGaussianMixtureModel
from distributions.CGaussianMixtureModel import generateRandomGMM
from distributions.CMultivariateNormal import CMultivariateNormal
from distributions.CRosenbrockPDF import CRosenbrock
from distributions.CRipple import CRipple
from distributions.CKernelDensity import CKernelDensity
from distributions.CNearestNeighbor import CNearestNeighbor

# SAMPLING METHODS
from sampling_methods.random_uniform import CRandomUniformSampling
from sampling_methods.tree_pyramid import CTreePyramidSampling
from sampling_methods.metropolis_hastings import CMetropolisHastings
from sampling_methods.rejection import CRejectionSampling
from sampling_methods.grid import CGridSampling
from sampling_methods.nested import CNestedSampling
from sampling_methods.multi_nested import CMultiNestedSampling

# UTILS FOR SAMPLING
from sampling_methods.base import uniform_sample_distribution
from sampling_methods.base import grid_sample_distribution
from sampling_methods.base import kl_divergence
from sampling_methods.base import bhattacharyya_distance

# PLOTTING FUNCTIONS
from utils.plot_utils import plot_grid_sampled_pdfs
from utils.plot_utils import plot_ellipsoids
from utils.plot_utils import plot_ellipsoids1D
from utils.plot_utils import plot_tpyramid_area
from utils.plot_utils import plot_tpyramid_volume
from utils.plot_utils import plot_grid_area
from utils.plot_utils import plot_grid_volume


if __name__ == "__main__":
    np.random.seed(3)
    make_plot = True
    save_plot = False
    kde_bw = 0.2                    # bandwidth for the KDE approximation
    sampling_eval_resolution = 0.05 # Resolution for the grid used to compute KL and Bhatacharayya metrics
    num_gaussians_gmm = 5           # Number of mixture components in the GMM model
    gmm_sigma_min = 0.05            # Miminum sigma value for the Normal family models
    gmm_sigma_max = 0.01            # Maximum sigma value for the Normal family models
    space_size = 1                  # Size of the domain for each dimension [0, n)

    if len(sys.argv) == 4:
        method = sys.argv[1]        # Sampling method used (see options in the comments below)
        distribution = sys.argv[2]  # Target PDF
        ndims = int(sys.argv[3])    # Number of dimensions
    else:
        # Select sampling method:
        # method = "BSPT"
        method = "reject"
        # method = "adaptive-reject"#TODO
        # method = "random"
        # method = "grid"
        # method = "nested"
        # method = "multi-nested"
        # method = "MCMC-MH"

        # Select target distribution:
        distribution = "gmm"
        # distribution = "rosenbrock"
        # distribution = "ripple"k
        # distribution = "normal"

        # Default dimensionality
        ndims = 2

    batch_samples = int(10 ** (ndims / 2))
    # batch_samples = 100
    max_samples = batch_samples * 100
    plot_sample_offset = 0
    print("Samples in batch: ", batch_samples)

    # Compute the limits of the space and set the origin at the center
    space_min = t_tensor([-space_size] * ndims)
    space_max = t_tensor([space_size]* ndims)
    origin = (space_min + space_max) / 2.0

    # Select the dimensions to marginalize (for visualization purposes)
    marginalize_dimensions = tuple(range(2,ndims))

    MCMC_proposal_dist = CMultivariateNormal(origin, np.diag(np.ones_like(space_max)) * 0.1)
    rejection_sampling_proposal_dist = CMultivariateNormal(origin, np.diag(np.ones_like(space_max)))
    if distribution == "gmm":
        target_dist = generateRandomGMM(space_min, space_max, num_gaussians_gmm, sigma_min=[gmm_sigma_min]*ndims, sigma_max=[gmm_sigma_max]*ndims)
    if distribution == "normal":
        target_dist = CMultivariateNormal(t_tensor([0,0]), t_tensor(np.diag([0.1,0.1])))
    elif distribution == "rosenbrock":
        target_dist = CRosenbrock(a=2, b=50.0)
    elif distribution == "ripple":
        target_dist = CRipple(amplitude=1.0, freq=20.0)

    grid_samples = 10
    samples_acc = np.random.uniform(space_min, space_max).reshape(-1,ndims)
    samples_logprob_acc = np.exp(target_dist.log_prob(samples_acc))

    if make_plot:
        # fig = plt.figure(figsize=(30,10))
        if ndims > 1:
            ax2 = plt.subplot(132, projection='3d')
            ax1 = plt.subplot(131, projection='3d', sharex=ax2, sharey=ax2, sharez=ax2)
            # ax3 = plt.subplot(133, projection='3d', sharex=ax2, sharey=ax2, sharez=ax2)
            ax3 = plt.subplot(133)
        else:
            fig, (ax1,ax3) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 2]})
            # ax1 = plt.subplot(131)
            ax2 = ax1
        # ax3 = plt.subplot(133)
        # ax4 = plt.figure().gca()
        # figManager = plt.get_current_fig_manager()
        # figManager.full_screen_toggle()
        plt.show(block=False)

    if method == "BSPT":
        sample_method = CTreePyramidSampling(space_min, space_max)
        if make_plot:
            sample_method.T.axis = ax1
    elif method == "random":
        sample_method = CRandomUniformSampling(space_min, space_max)
    elif method == "MCMC-MH":
        sample_method = CMetropolisHastings(space_min, space_max, MCMC_proposal_dist)
    elif method == "reject":
        sample_method = CRejectionSampling(space_min, space_max, rejection_sampling_proposal_dist, scaling=1)
    elif method == "grid":
        sample_method = CGridSampling(space_min, space_max)
    elif method == "nested":
        sample_method = CNestedSampling(space_min, space_max, MCMC_proposal_dist, num_points=30)
    elif method == "multi-nested":
        sample_method = CMultiNestedSampling(space_min, space_max, num_points=30)
    else:
        raise NotImplementedError

    kl_div = 0
    bhattacharyya_dist = 0
    n_iter = 0
    kl_div_hist = []
    n_samples_hist = []
    timings = {"sampling_step":0, "kde":0, "grid_sample_kde":0, "grid_sample_target":0, "plot":0, "kl_div":0, "bhattacharyya":0}

    print("Computing ground truth sampled distribution...")
    eval_samples, p_samples_logprob, dims, shape = grid_sample_distribution(target_dist, space_min, space_max, resolution=sampling_eval_resolution)
    p_samples_prob = np.exp(p_samples_logprob)
    p_samples_weight = p_samples_prob / p_samples_prob.sum()
    gt_expected_value_weighted = eval_samples * p_samples_weight.reshape(-1,1)
    gt_expected_value = gt_expected_value_weighted.sum(axis=0)
    gt_expected_variance = gt_expected_value_weighted.var(axis=0)
    # eval_samples, p_samples_prob = uniform_sample_distribution(target_dist, space_min, space_max, nsamples=sampling_eval_samples)
    print("DONE!")

    # Make a separate plot with the GT distribution
    # fig2 = plt.figure(figsize=(20,30))
    # ax_tmp = plt.subplot(111, projection='3d')
    # plot_grid_sampled_pdfs(ax_tmp, dims, np.exp(p_samples_prob), shape=(len(dims[0]), len(dims[1])), marginalize_axes=marginalize_dimensions)

    while len(samples_logprob_acc) < max_samples:
        # Compute expected value
        expected_value = samples_acc.mean(axis=0)
        expected_value_weighted = (samples_acc * np.exp(samples_logprob_acc.reshape(-1,1))).mean()
        # Compute expected value variance
        expected_value_var = samples_acc.var(axis=0)
        expected_value_weighted_var = (samples_acc * np.exp(samples_logprob_acc.reshape(-1,1))).var()

        if make_plot and ndims == 1:
            ax1.axvline(expected_value, c='r', label="approximate EV")
            ax1.axvline(gt_expected_value, c='b', label="target EV")


        tic = time.time()
        if method=="grid":
            samples_acc, samples_logprob_acc = sample_method.sample_with_likelihood(pdf=target_dist, n_samples=grid_samples)
            grid_samples = batch_samples + grid_samples
        else:
            samples, samples_logprob = sample_method.sample_with_likelihood(pdf=target_dist, n_samples=batch_samples)
            samples_acc = np.vstack((samples_acc, samples))
            samples_logprob_acc = np.hstack((samples_logprob_acc, samples_logprob))

        timings["sampling_step"] = time.time() - tic

        # Obtain the approximate density for the samples
        tic = time.time()
        approximate_pdf2 = CNearestNeighbor(samples_acc, samples_logprob_acc)
        approximate_pdf = CKernelDensity(samples_acc, np.exp(samples_logprob_acc), bw=0.1)
        timings["approximate_pdf_build"] = time.time() - tic
        tic = time.time()
        kde_samples_prob = approximate_pdf.log_prob(eval_samples)
        nn_samples_prob = approximate_pdf2.log_prob(eval_samples)
        timings["approximate_pdf_eval"] = time.time() - tic

        if make_plot:
            tic = time.time()
            if ndims==1:
                ax1.plot(eval_samples, np.exp(p_samples_logprob), "-", label="target PDF")     # Plot sampled ground truth probability
                # ax1.plot(eval_samples, np.exp(kde_samples_prob), ".", c="r")    # Plot sampled approximate distribution
            else:
                # plot_sampled_pdfs(ax1, eval_samples, np.exp(p_samples_prob), marginalize_axes = marginalize_dimensions)
                # plot_sampled_pdfs(ax2, eval_samples, np.exp(kde_samples_prob), marginalize_axes=marginalize_dimensions)
                plot_grid_sampled_pdfs(ax1, dims, np.exp(p_samples_logprob), shape=(len(dims[0]), len(dims[1])), marginalize_axes = marginalize_dimensions)
                # plot_grid_sampled_pdfs(ax2, dims, np.exp(kde_samples_prob), shape=(len(dims[0]),len(dims[1])), marginalize_axes=marginalize_dimensions)
                # plot_grid_sampled_pdfs(ax3, dims, np.exp(nn_samples_prob), shape=(len(dims[0]), len(dims[1])),marginalize_axes=marginalize_dimensions)
            timings["plot"] = time.time() - tic

        tic = time.time()
        kl_div = kl_divergence(np.exp(p_samples_logprob), np.exp(kde_samples_prob))
        kl_div2 = kl_divergence(np.exp(p_samples_logprob), np.exp(nn_samples_prob))
        timings["kl_div"] = time.time() - tic

        tic = time.time()
        bhattacharyya_dist = bhattacharyya_distance(np.exp(p_samples_logprob), np.exp(kde_samples_prob))
        bhattacharyya_dist2 = bhattacharyya_distance(np.exp(p_samples_logprob), np.exp(nn_samples_prob))
        timings["bhattacharyya"] = time.time() - tic

        text_info = "%s_%03d.png #dims: %d #samples: %d KL: %f bhattacharyya: %f" % (method, n_iter, ndims, len(samples_logprob_acc), kl_div, bhattacharyya_dist)

        if make_plot:
            if method.find("MCMC") != -1 and ndims > 1:
                ax2.plot(samples_acc[:, 0], samples_acc[:, 1], np.ones(len(samples_acc)) * plot_sample_offset, marker=".", alpha=0.4)
                ax1.plot(samples_acc[:, 0], samples_acc[:, 1], np.ones(len(samples_acc)) * plot_sample_offset, marker=".", alpha=0.4)
            elif method.find("nested") != -1:
                if ndims > 1:
                    ax2.scatter(sample_method.live_points[:, 0], sample_method.live_points[:, 1], np.ones(len(sample_method.live_points)) * plot_sample_offset, marker="o", alpha=1, edgecolor="g",c="g")
                    ax2.scatter(samples_acc[:, 0], samples_acc[:, 1], np.ones(len(samples_acc)) * plot_sample_offset, marker=2, alpha=1)
                else:
                    ax2.plot(sample_method.live_points[:, 0], np.zeros(len(sample_method.live_points)), marker="o", alpha=1, c="g")
                    ax2.plot(samples_acc[:, 0], np.zeros(len(samples_acc)), marker=2, alpha=1, c="r")
            elif ndims > 1:
                ax2.scatter(samples_acc[:,0], samples_acc[:,1], np.ones(len(samples_acc)) * plot_sample_offset, marker=".", alpha=1, c="w", zorder=10)
            else:
                ax2.plot(samples_acc[:,0], np.zeros(len(samples_acc)), marker=2, alpha=1, c="r")

            if method.find("multi") != -1:
                if ndims > 1:
                    plot_ellipsoids(ax2, sample_method.ellipsoids, sample_method.live_points)
                else:
                    plot_ellipsoids1D(ax2, sample_method.ellipsoids, sample_method.live_points)

            if method == "BSPT":
                sample_method.T.plot(ax3)
                if ndims == 1:
                    plot_tpyramid_area(ax1, sample_method.T)
                    #######################
                    ## Temporary code to make the legend
                    # rect = patches.Rectangle([0, 0], 0.01, 0.01, linewidth=0, linestyle="--", alpha=0.4, color=cm.cool(.5), label="approximation")
                    # ax1.add_patch(rect)
                    # rect = patches.Rectangle([0, 0], 0.01, 0.01, linewidth=3, linestyle="--", fill=False, alpha=1.0, color='g', label="expanded nodes")
                    # ax1.add_patch(rect)
                    #######################
                else:
                    plot_tpyramid_volume(ax2, sample_method.T)

            if method == "grid":
                if ndims == 1:
                    plot_grid_area(ax1, samples_acc, np.exp(samples_logprob_acc), sample_method.resolution)
                    text_info = text_info + " integral: %f" % sample_method.integral
                else:
                    plot_grid_volume(ax2, samples_acc, np.exp(samples_logprob_acc), sample_method.resolution)
                    text_info = text_info + " integral: %f" % sample_method.integral

            # ax1.set_xlim(space_min[0], space_max[0])
            # ax1.set_ylim(space_min[1], space_max[1])
            # ax1.set_zlim(0, 0.5)

            ev_mse = ((gt_expected_value-expected_value) * (gt_expected_value-expected_value)).sum()
            text_info = " ev_mse: %.4f " % ev_mse + text_info
            plt.suptitle(text_info)
            kl_div_hist.append(kl_div)
            n_samples_hist.append(len(samples_logprob_acc))
            # ax3.plot(n_samples_hist,kl_div_hist, c="g")
            # ax1.set_title("Target Distribution: %s" % distribution)
            # ax2.set_title("KDE approximation. KL: %f B: %f" % (kl_div, bhattacharyya_dist))
            # ax3.set_title("NN approximation. KL: %f B: %f" % (kl_div2, bhattacharyya_dist2))
            ax1.autoscale(True)
            # ax2.autoscale(True)
            # ax2.set_title("KDE distribution. $\omega = %3.2f$" % kde_bw)
            if save_plot:
                plt.savefig("%s_%s_%03d.png" % (method,distribution,n_iter))
            plt.pause(0.01)
            ax1.legend()
            ax1.cla()
            ax2.cla()
            ax3.cla()

        if method == "BSPT":
            if ndims == 1:
                text_info = text_info + " integral: %f" % sample_method.integral
            else:
                text_info = text_info + " integral: %f" % sample_method.integral

        print(text_info)
        file = open("res_%s_%s_%d.txt" % (method, distribution, ndims), mode='a')
        file.write("%d %f %f\n" % (len(samples_logprob_acc), kl_div, bhattacharyya_dist))
        file.close()

        print(timings)

        time.sleep(0.05)
        n_iter = n_iter + 1
