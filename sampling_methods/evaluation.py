import numpy as np
import time
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.stats import entropy

from sampling_methods.base import t_tensor
from sampling_methods.base import uniform_sample_distribution
from sampling_methods.base import grid_sample_distribution
from distributions.CKernelDensity import CKernelDensity
from distributions.CNearestNeighbor import CNearestNeighbor
from utils.plot_utils import plot_grid_sampled_pdfs
from utils.plot_utils import plot_tpyramid_area


def log_print(text, file, mode='a+'):
    with open(file, mode=mode) as f:
        f.write(text + "\n")
        print(text)


def bhattacharyya_distance(p_samples_prob, q_samples_prob):
    res = - np.log(np.sum(np.sqrt(p_samples_prob * q_samples_prob)))
    return res


def kl_divergence_components(p_samples_prob, q_samples_prob):
    p_samples_prob_norm = p_samples_prob / np.sum(p_samples_prob)
    q_samples_prob_norm = q_samples_prob / np.sum(q_samples_prob)
    res = p_samples_prob_norm * np.log(p_samples_prob_norm / q_samples_prob_norm)
    return res


def kl_divergence(p_samples_prob, q_samples_prob):
    # p_samples_prob_norm = p_samples_prob / np.sum(p_samples_prob)
    # q_samples_prob_norm = q_samples_prob / np.sum(q_samples_prob)
    # res = (p_samples_prob_norm * np.log(p_samples_prob_norm / q_samples_prob_norm)).sum()

    res = entropy(pk=p_samples_prob, qk=q_samples_prob)
    return res


def evaluate_samples(samples, samples_logprob, target_dist, space_min, space_max, sampling_eval_samples=1000):

    # Generate random samples and obtain its true density from the target distribution
    eval_samples, p_samples_logprob = uniform_sample_distribution(target_dist, space_min, space_max, nsamples=sampling_eval_samples)

    # Approximate the target density with the input samples using Kernel Density and Nearest Neighbor approximations
    approximate_pdf = CKernelDensity(samples, np.exp(samples_logprob), bw=0.1)
    approximate_pdf2 = CNearestNeighbor(samples, samples_logprob)

    # Compute the approximated densities
    kde_samples_logprob = approximate_pdf.log_prob(eval_samples)
    nn_samples_logprob = approximate_pdf2.log_prob(eval_samples)

    # Compute the kl divergence of the densities obtained from both sample sets
    kl_div_kde = kl_divergence(np.exp(p_samples_logprob), np.exp(kde_samples_logprob))
    kl_div_nn = kl_divergence(np.exp(p_samples_logprob), np.exp(nn_samples_logprob))

    # Compute the bhattacharyya distance of the two sample sets
    bhattacharyya_dist_kde = bhattacharyya_distance(np.exp(p_samples_logprob), np.exp(kde_samples_logprob))
    bhattacharyya_dist_nn = bhattacharyya_distance(np.exp(p_samples_logprob), np.exp(nn_samples_logprob))

    # Compute the expected value mean squared error
    expected_value = (samples * np.exp(samples_logprob.reshape(-1, 1))).sum(axis=0)
    gt_expected_value = (eval_samples * np.exp(p_samples_logprob.reshape(-1, 1))).sum(axis=0)
    ev_mse = ((gt_expected_value - expected_value) * (gt_expected_value - expected_value)).sum()

    # kl_div_kde = 0
    # bhattacharyya_dist_kde = 0

    return kl_div_kde, kl_div_nn, bhattacharyya_dist_kde, bhattacharyya_dist_nn, ev_mse


def evaluate_method(ndims, space_size, target_dist, sampling_method, max_samples, sampling_eval_samples, debug=True, filename=None, max_sampling_time=600):
    batch_samples = int(ndims ** 2)  # Number of samples per batch of samples
    # batch_samples = 10
    space_min = t_tensor([-space_size] * ndims)
    space_max = t_tensor([space_size] * ndims)

    samples_acc = np.random.uniform(space_min, space_max).reshape(-1, ndims)
    samples_logprob_acc = np.exp(target_dist.log_prob(samples_acc))

    if debug:
        pts = []
        if ndims == 1:
            fig = plt.figure(figsize=(10, 8))
            ax = plt.subplot(111)
            plt.hold(True)
            plt.show(block=False)
            resolution = 0.02
            x = np.linspace(space_min, space_max, int((space_max-space_min)/resolution)).reshape(-1,1)
            y = np.exp(target_dist.log_prob(x))
            ax.plot(x, y, "-b", alpha=0.2)
            plt.xlim(space_min, space_max)
            plt.ylim(0, 2)

        elif ndims == 2:
            fig = plt.figure(figsize=(10, 8))
            ax = plt.subplot(111, projection='3d')
            plt.hold(True)
            plt.show(block=False)
            grid, log_prob, dims, shape = grid_sample_distribution(target_dist, space_min, space_max, resolution=0.02)
            plot_grid_sampled_pdfs(ax, dims, np.exp(log_prob), shape=shape, alpha=0.4)

    # Perform sampling
    sampling_time = 0
    sampling_method.reset()
    pts = []
    n_samples = batch_samples
    while len(samples_acc) < max_samples:
        t_ini = time.time()

        samples_acc, samples_logprob_acc = sampling_method.importance_sample(target_d=target_dist,
                                                                             n_samples=n_samples,
                                                                             timeout=max_sampling_time - sampling_time,
                                                                             resampling="ancestral")

        n_samples = len(samples_acc) + batch_samples

        sampling_time += time.time()-t_ini

        if filename is not None:
            t_ini = time.time()
            [kl_div_kde, kl_div_nn, bhattacharyya_dist_kde, bhattacharyya_dist_nn, ev_mse] = \
                evaluate_samples(samples_acc, samples_logprob_acc, target_dist, space_min, space_max, sampling_eval_samples)
            print("Evaluation time: %3.3f nsamples: %d samples/sec: %3.3f" % (time.time() - t_ini, len(samples_acc), len(samples_acc) / (time.time() - t_ini)))

            log_print("  %02d %08d %7.4f %.5f %.5f %.5f %6.3f %s %d %s" % (
            ndims, max_samples, kl_div_kde, bhattacharyya_dist_kde, kl_div_nn, bhattacharyya_dist_nn, sampling_time,
            sampling_method.name, len(samples_acc), target_dist.name), file=filename)

            if debug:
                plt.suptitle("%s | #smpl: %d | KL: %3.3f BHT: %3.3f" % (sampling_method.name, n_samples, kl_div_kde, bhattacharyya_dist_kde))

        if sampling_time > max_sampling_time:
            break

        # DEBUG CODE HERE
        if debug:
            # Remove previous points
            for element in pts:
                element.remove()
            pts.clear()
            if ndims == 1:
                pts.extend(plot_tpyramid_area(ax, sampling_method.T))
                pts.extend(ax.plot(samples_acc, samples_logprob_acc, "r."))
                pts.extend(ax.plot(samples_acc, np.zeros_like(samples_logprob_acc), "r|"))
                plt.pause(0.01)
            if ndims == 2:
                pts.append(ax.scatter(samples_acc[:, 0], samples_acc[:, 1], -1, label="samples", c="r", marker="o"))
                plt.pause(0.01)

    res = evaluate_samples(samples_acc, samples_logprob_acc, target_dist, space_min, space_max, sampling_eval_samples)
    res = list(res)
    res.append(len(samples_acc))
    if debug:
        if ndims == 1:
            plt.suptitle("%s | #smpl: %d | KL: %3.3f BHT: %3.3f" % (sampling_method.name, max_samples, res[0], res[2]))
            plt.pause(0.01)
            plt.close()

    return res
