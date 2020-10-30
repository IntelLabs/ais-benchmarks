import numpy as np
import time
import cProfile
import pstats

from sampling_methods.base import t_tensor
from metrics.divergences import CKLDivergence
from metrics.performance import CMemoryUsage
from metrics.performance import CElapsedTime
from utils.misc import time_to_hms


def log_print(text, file, mode='a+'):
    with open(file, mode=mode) as f:
        f.write(text + "\n")


def write_results(results, method, target, nsamples, file):
    with open(file, mode='a+') as f:
        text = "%02d %04d " % (target.dims, nsamples)
        text += " ".join(["%7.5f" % val for val in results.values()])
        text += "%7.4f %s %s %5.3f %d %d %d" % (method.get_NESS(), method.name, target.name,
                                                method.get_acceptance_rate(), method.num_proposal_samples,
                                                method.num_proposal_evals, method.num_target_evals)
        f.write(text + "\n")


def evaluate_method(ndims, target_dist, sampling_method, max_samples, sampling_eval_samples,
                    metrics=("NESS", "JSD", "T"), rseed=0, n_reps=10, batch_size=16,
                    debug=True, filename=None, max_sampling_time=600, profile=False):
    """
    Parameters
    ----------
    ndims : int
        Number of dimensions of the target distribution

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

    # Create metrics instances
    metrics_eval = list()
    for m in metrics:
        if m == "KLD":
            metrics_eval.append(CKLDivergence())
        elif m == "JSD":
            pass
        elif m == "NESS":
            pass
        elif m == "EV_MSE":
            pass
        elif m == "T":
            metrics_eval.append(CElapsedTime())
        elif m == "MEM":
            metrics_eval.append(CMemoryUsage())

    # Repeat the experiment n_reps times
    for nexp in range(n_reps):

        # Initialize sampling variables
        t_start = time.time()
        sampling_time = 0
        sampling_method.reset()
        samples_acc = t_tensor([])
        n_samples = batch_size
        [m.reset() for m in metrics_eval]

        # Perform importance sampling until the desired number of samples is obtained
        while len(samples_acc) < max_samples:

            # Obtain experiment execution runtime
            h, m, s = time_to_hms(time.time() - t_start)

            # Display partial sampling experiment statistics
            text_display = "%02d/%02d | %s | %s | %dD | #s: %.1fk | %5.1f%% | t: %02dh %02dm %4.1fs | " % (
                nexp + 1, n_reps, sampling_method.name, target_dist.name, ndims, len(samples_acc)/1000.0,
                (len(samples_acc)/max_samples)*100, h, m, s)

            text_display += " | ".join(["%s: %7.5f" % (m.name, m.value) for m in metrics_eval])
            print(text_display, end="\r", flush=True)

            # Initialize metrics before running the evaluated code
            [m.pre() for m in metrics_eval]

            # Perform importance sampling operation to generate a batch of samples and compute the time taken
            samples_acc, _ = sampling_method.importance_sample(target_d=target_dist, n_samples=n_samples,
                                                               timeout=max_sampling_time - sampling_time)

            # Compute metrics right after the sampling operation
            [m.post() for m in metrics_eval]

            # Sampling methods generate a desired number of samples and maintain the state. Therefore it is not possible
            # to ask the sampling algos for batch_size samples every time, thus in order to have partial results during
            # the sampling experiments, the number of samples to generate is always increase by batch size. This lets
            # us measure partial results in increments of batch_size samples.
            n_samples = len(samples_acc) + batch_size

            # Perform sampling evaluation and store results to file
            if filename is not None:
                results = dict()

                # Compute desired metrics
                for m in metrics_eval:
                    val = m.compute(p=target_dist, q=sampling_method, nsamples=sampling_eval_samples)
                    results[m.name] = val

                # Write metric results to file
                write_results(results=results, method=sampling_method, target=target_dist,
                              nsamples=len(samples_acc), file=filename)

            # Some sampling algorithms may take too long to generate the desired number of samples, the timeout ensures
            # the operation will end if the specified timeout is reached.
            if sampling_time > max_sampling_time:
                break

        # Obtain experiment execution runtime
        h, m, s = time_to_hms(time.time() - t_start)

        # Print final experiment statistics
        text_display = "%02d/%02d | %s | %s | %dD | #s: %.1fk | %5.1f%% | t: %02dh %02dm %4.1fs | " % (
            nexp + 1, n_reps, sampling_method.name, target_dist.name, ndims, len(samples_acc) / 1000.0,
            (len(samples_acc) / max_samples) * 100, h, m, s)

        text_display += " | ".join(["%s: %7.5f" % (m.name, m.value) for m in metrics_eval])
        print(text_display, end="\n", flush=True)
        #
        # print(
        #     "Exp %02d/%02d || %s || %s || %dD || #s: %.1fk || %5.1f%% || t: %02dh %02dm %4.1fs || mem: %.1fMB" % (
        #         nexp + 1, n_reps, sampling_method.name, target_dist.name, ndims, len(samples_acc) / 1000.0,
        #         (len(samples_acc) / max_samples) * 100, h, m, s, mem_used), end="\n", flush=True)

    # Stop profiling tools and save results
    if profile:
        profiler.disable()
        ps = pstats.Stats(profiler)
        ps.sort_stats("cumtime")
        ps.dump_stats(filename + ".profile")
        ps.print_stats()

    # TODO: Print experiment metric stats

    # Return sampling visualization data
    return sampling_method.get_viz_frames()
