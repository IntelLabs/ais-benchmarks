import numpy as np
from scipy.stats import entropy

from sampling_methods.base import t_tensor
from sampling_methods.base import uniform_sample_distribution
from distributions.CKernelDensity import CKernelDensity
from distributions.CNearestNeighbor import CNearestNeighbor


def bhattacharyya_distance(p_samples_prob, q_samples_prob):
    res = - np.log(np.sum(np.sqrt(p_samples_prob * q_samples_prob)))
    return res


def kl_divergence(p_samples_prob, q_samples_prob):
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

    return kl_div_kde, kl_div_nn, bhattacharyya_dist_kde, bhattacharyya_dist_nn, ev_mse


def evaluate_method(ndims, space_size, target_dist, sampling_method, max_samples, sampling_eval_samples):
    batch_samples = int(10 ** (ndims / 2))  # Number of samples per batch of samples
    space_min = t_tensor([-space_size] * ndims)
    space_max = t_tensor([space_size] * ndims)

    samples_acc = np.random.uniform(space_min, space_max).reshape(-1,ndims)
    samples_logprob_acc = np.exp(target_dist.log_prob(samples_acc))

    # Perform sampling
    while len(samples_acc) < max_samples:
        samples, samples_logprob = sampling_method.sample_with_likelihood(pdf=target_dist, n_samples=batch_samples)
        samples_acc = np.vstack((samples_acc, samples))
        samples_logprob_acc = np.hstack((samples_logprob_acc, samples_logprob))

    return evaluate_samples(samples_acc, samples_logprob_acc, target_dist, space_min, space_max, sampling_eval_samples)
