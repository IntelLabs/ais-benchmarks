import numpy as np
import time
import cProfile
import pstats
import tracemalloc

from scipy.stats import entropy

from sampling_methods.base import t_tensor
from sampling_methods.base import uniform_sample_distribution
from distributions.nonparametric.CKernelDensity import CKernelDensity
from utils.misc import time_to_hms


def log_print(text, file, mode='a+'):
    with open(file, mode=mode) as f:
        f.write(text + "\n")


def bhattacharyya_distance(p_samples_prob, q_samples_prob):
    res = - np.log(np.sum(np.sqrt(p_samples_prob * q_samples_prob)))
    return res


def kl_divergence_components_logprob(p_samples_logprob, q_samples_logprob):
    p_samples_prob = np.exp(p_samples_logprob - np.max(p_samples_logprob))
    q_samples_prob = np.exp(q_samples_logprob - np.max(q_samples_logprob))
    res = kl_divergence_components(p_samples_prob, q_samples_prob)
    return res


def kl_divergence_components(p_samples_prob, q_samples_prob):
    p_samples_prob_norm = p_samples_prob / np.sum(p_samples_prob)
    q_samples_prob_norm = q_samples_prob / np.sum(q_samples_prob)

    # handle zero probabilities
    inliers_p = np.logical_and(q_samples_prob_norm > 0, p_samples_prob_norm > 0)

    res = p_samples_prob_norm * (np.log(p_samples_prob_norm) - np.log(q_samples_prob_norm))
    res[np.logical_not(inliers_p)] = 0
    return res


def kl_divergence_logprob(p_samples_logprob, q_samples_logprob):
    p_samples_prob = np.exp(p_samples_logprob - np.max(p_samples_logprob))
    q_samples_prob = np.exp(q_samples_logprob - np.max(q_samples_logprob))
    res = kl_divergence(p_samples_prob, q_samples_prob)

    # res = entropy(pk=p_samples_prob, qk=q_samples_prob)
    return res


def kl_divergence(p_samples_prob, q_samples_prob):
    # p_samples_prob_norm = p_samples_prob / np.sum(p_samples_prob)
    # q_samples_prob_norm = q_samples_prob / np.sum(q_samples_prob)
    # res = (p_samples_prob_norm * np.log(p_samples_prob_norm / q_samples_prob_norm)).sum()

    res = entropy(pk=p_samples_prob, qk=q_samples_prob)
    return res


def js_divergence(p_samples_prob, q_samples_prob):
    m_samples_prob = 0.5 * (p_samples_prob + q_samples_prob)
    res = 0.5 * kl_divergence_components(p_samples_prob, m_samples_prob) + 0.5 * kl_divergence_components(q_samples_prob, m_samples_prob)
    return res


def js_divergence_logprob(p_samples_logprob, q_samples_logprob):
    p_samples_prob = np.exp(p_samples_logprob - np.max(p_samples_logprob))
    q_samples_prob = np.exp(q_samples_logprob - np.max(q_samples_logprob))
    m_samples_prob = 0.5 * (p_samples_prob + q_samples_prob)
    res = 0.5 * kl_divergence_components(p_samples_prob, m_samples_prob) + \
          0.5 * kl_divergence_components(q_samples_prob, m_samples_prob)

    return res


def evaluate_proposal(proposal_dist, target_dist, space_min, space_max, sampling_eval_samples=1000):
    # Generate random samples and obtain its true density from the target distribution
    eval_samples, p_samples_logprob = uniform_sample_distribution(target_dist, space_min, space_max, nsamples=sampling_eval_samples)

    # Obtain the density at the sampled points from the proposal distribution
    q_samples_logprob = proposal_dist.log_prob(eval_samples)

    # Compute the empirical Jensen-Shannon Divergence
    js_div_comps = js_divergence_logprob(p_samples_logprob.flatten(), q_samples_logprob)
    js_div = np.sum(js_div_comps)
    # print(list(js_div_comps))

    # Compute the empirical Bhattacharyya distance
    bhattacharyya_dist = bhattacharyya_distance(np.exp(p_samples_logprob.flatten()), np.exp(q_samples_logprob))

    # Compute the empirical expected value mean squared error
    expected_value = (eval_samples * np.exp(q_samples_logprob.reshape(-1, 1))).sum(axis=0)
    gt_expected_value = (eval_samples * np.exp(p_samples_logprob.reshape(-1, 1))).sum(axis=0)
    ev_mse = ((gt_expected_value - expected_value) * (gt_expected_value - expected_value)).sum()

    return js_div, bhattacharyya_dist, ev_mse / sampling_eval_samples


def evaluate_samples(samples, samples_logprob, target_dist, space_min, space_max, sampling_eval_samples=1000):

    # Approximate the target density with the input samples using Kernel Density and Nearest Neighbor approximations
    approximate_pdf = CKernelDensity(samples, np.exp(samples_logprob), bw=0.1)

    return evaluate_proposal(approximate_pdf, target_dist, space_min, space_max, sampling_eval_samples)


def evaluate_method(ndims, support, target_dist, sampling_method, max_samples, sampling_eval_samples,
                    metrics=["NESS", "JSD", "T"], rseed=0, n_reps=10, batch_size=16,
                    debug=True, filename=None, max_sampling_time=600, profile=False):
    """
    Parameters
    ----------
    ndims : int
        Number of dimensions of the target distribution

    support : array_like
        Support limits. support[0] contains the lower limits and support[1] the upper limits for the support of the
        target distribution. The support limits are used to generate the samples for the approximate JSD and other
        MC approximations for the metrics. If the target has infinite support, a reasonable support containing most
        of the probability mass needs to be specified.

    target_dist : CDistribution
        Target distribution under evaluation. This can be an object derived from CDistribution or any other object that
        implements prob(x) and log_prob(x) methods where x is a batch of samples.

    sampling_method : CSamplingMethod
        Object derived from CSamplingMethod that represents/implements the method to be evaluated.

    max_samples : int
        Number of samples to obtain with the sampling algorithm to conclude the evaluation.

    sampling_eval_samples : int
        Number of Monte Carlo samples used to approximate the measured metrics.

    metrics : list[str]
        List of strings that specify the metrics to be computed. Default: ["NESS", "JSD", "T"]

    rseed: int
        Random seed to be used. If none is specified, the random generator state is not altered. Default: None

    n_reps: int
        Number of times to repeat each experiment. Default: 10.

    batch_size: int
        Number of samples to be generated before computing the partial metrics. Default: 16.

    debug: bool
        Flag to enable console debug messages and other debugging visualizations. Default: False.

    filename : str
        Path to store the experiment results, use None to disable result saving. Default: None.

    max_sampling_time : float
        Time limit in seconds for the generation of max_samples using the specified sampling_method. When the timeout
        is reached the sampling process will be aborted and the partial results written to disk. Default: 600.0

    profile : bool
        Flag to enable profiling results. Consider using when debugging or analyzing an algorithm as the usage of the
        profiler might impact the performance. Will use the provided filename to name the profiling file results.
        Default: False.

    Returns
    -------
        List of CVisual objects. The list describes the sequence in which the visual objects must be shown, the list
        can contain multiple CVisual objects with the same id. That should be interpreted as the same object updating
        its visual representation, therefore it can be used by the visualization to replace the previous displayed
        object by the new one or use an animation to display the transformation.

    Raises
    -------
    """

    # Set the random seed if specified, important for reproducibility
    if rseed is not None:
        np.random.seed(rseed)

    # Start profiling tools
    if profile:
        profiler = cProfile.Profile()
        profiler.enable()

    # Start memory profiling
    tracemalloc.start()

    # Repeat the experiment n_reps times
    for nexp in range(n_reps):

        # Initialize sampling variables
        t_start = time.time()
        sampling_time = 0
        sampling_method.reset()
        samples_acc = t_tensor([])
        n_samples = batch_size

        # Perform importance sampling until the desired number of samples is obtained
        while len(samples_acc) < max_samples:

            # Compute current memory usage in MB
            mem_used = tracemalloc.get_tracemalloc_memory()/(1024.0*1024.0)

            # Obtain experiment execution runtime
            h, m, s = time_to_hms(time.time() - t_start)

            # Display partial sampling experiment statistics
            print("Exp %02d/%02d || %s || %s || %dD || #s: %.1fk || %5.1f%% || t: %02dh %02dm %4.1fs || mem: %.1fMB" % (
                nexp + 1, n_reps, sampling_method.name, target_dist.name, ndims, len(samples_acc)/1000.0,
                (len(samples_acc)/max_samples)*100, h, m, s, mem_used), end="\r", flush=True)

            # Perform importance sampling operation to generate a batch of samples and compute the time taken
            t_ini = time.time()
            samples_acc, _ = sampling_method.importance_sample(target_d=target_dist, n_samples=n_samples,
                                                               timeout=max_sampling_time - sampling_time)
            sampling_time += time.time()-t_ini

            # Sampling methods generate a desired number of samples and maintain the state. Therefore it is not possible
            # to ask the sampling algos for batch_size samples every time, thus in order to have partial results during
            # the sampling experiments, the number of samples to generate is always increase by batch size. This lets
            # us measure partial results in increments of batch_size samples.
            n_samples = len(samples_acc) + batch_size

            # Perform sampling evaluation and store results to file
            if filename is not None:
                # TODO: Compute only desired metrics
                [js_div, bhattacharyya_dist, ev_mse] = \
                    evaluate_proposal(sampling_method, target_dist, support[0], support[1], sampling_eval_samples)

                log_print("%02d %04d %7.5f %7.5f %8.5f %7.5f %7.4f %s %s %5.3f %d %d %d %f7.2" % (
                    ndims, len(samples_acc), js_div, bhattacharyya_dist, ev_mse, sampling_method.get_NESS(), sampling_time,
                    sampling_method.name, target_dist.name, sampling_method.get_acceptance_rate(),
                    sampling_method.num_proposal_samples, sampling_method.num_proposal_evals, sampling_method.num_target_evals, mem_used), file=filename)

            # Some sampling algorithms may take too long to generate the desired number of samples, the timeout ensures
            # the operation will end if the specified timeout is reached.
            if sampling_time > max_sampling_time:
                break

        # Obtain experiment execution runtime
        h, m, s = time_to_hms(time.time() - t_start)

        # Print final experiment statistics
        print(
            "Exp %02d/%02d || %s || %s || %dD || #s: %.1fk || %5.1f%% || t: %02dh %02dm %4.1fs || mem: %.1fMB" % (
                nexp + 1, n_reps, sampling_method.name, target_dist.name, ndims, len(samples_acc) / 1000.0,
                (len(samples_acc) / max_samples) * 100, h, m, s, mem_used), end="\n", flush=True)

    # Stop profiling tools and save results
    tracemalloc.stop()
    if profile:
        profiler.disable()
        ps = pstats.Stats(profiler)
        ps.sort_stats("cumtime")
        ps.dump_stats(filename + ".profile")
        ps.print_stats()

    # Return sampling visualization data
    return sampling_method.get_viz_frames()
