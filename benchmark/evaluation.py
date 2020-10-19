import numpy as np
import time
from matplotlib import pyplot as plt

from scipy.stats import entropy

from sampling_methods.base import t_tensor
from sampling_methods.base import uniform_sample_distribution
from sampling_methods.base import grid_sample_distribution
from distributions.nonparametric.CKernelDensity import CKernelDensity
from utils.plot_utils import plot_grid_sampled_pdfs
from utils.plot_utils import plot_pdf
from utils.video_writer import CVideoWriter
from utils.misc import time_to_hms


def log_print(text, file, mode='a+'):
    with open(file, mode=mode) as f:
        f.write(text + "\n")
        #print(text)


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


def evaluate_method(ndims, space_size, target_dist, sampling_method, max_samples, sampling_eval_samples,
                    metrics=["NESS", "JSD", "T"], rseed=0, n_reps=10, batch_size=1,
                    debug=True, filename=None, max_sampling_time=600, videofile=None):

    np.random.seed(rseed)
    batch_samples = batch_size
    space_min = t_tensor([-space_size] * ndims)
    space_max = t_tensor([space_size] * ndims)

    samples_acc = np.random.uniform(space_min, space_max).reshape(-1, ndims)
    samples_logprob_acc = target_dist.log_prob(samples_acc)

    # Frame collection for the videofile
    if videofile is not None:
        vid_writer = CVideoWriter(videofile, fps=3)

    if debug:
        pts = []
        if ndims == 1:
            fig = plt.figure(figsize=(10, 8))
            ax = plt.subplot(111)
            # plt.hold(True)
            plt.show(block=False)
            plot_pdf(ax, target_dist, space_min, space_max, alpha=1.0, options="b-", resolution=0.01, label="$\pi(x)$")

            plt.xlim(space_min, space_max)
            plt.ylim(0, ax.get_ylim()[1])

        elif ndims == 2:
            fig = plt.figure(figsize=(10, 8))
            ax = plt.subplot(111)
            # ax = plt.subplot(111, projection='3d')
            # plt.hold(True)
            plt.show(block=False)

            grid, log_prob, dims, shape = grid_sample_distribution(target_dist, space_min, space_max, resolution=0.02)
            plot_grid_sampled_pdfs(ax, dims, np.exp(log_prob), shape=shape, alpha=1, label="$\pi(x)$", cmap='gray', linestyles='dashed')

            plt.xlim(space_min[0], space_max[0])
            plt.ylim(space_min[1], space_max[1])
            # ax.set_zlim(ax.get_zlim())

    for nexp in range(n_reps):
        # print("Experiment %d/%d || method: %s || dist: %s || dims: %d " % (nexp+1, n_reps, sampling_method.name, target_dist.name, ndims), end="", flush=True)
        t_start = time.time()

        # Perform sampling
        sampling_time = 0
        sampling_method.reset()
        samples_acc = t_tensor([])
        # pts = []
        n_samples = batch_samples
        while len(samples_acc) < max_samples:
            h, m, s = time_to_hms(time.time()-t_start)
            print("Exp %02d/%02d || %s || dist: %s || dims: %d || #smpls: %.1fk || %5.1f%% || t: %02dh %02dm %4.1fs" % (
                nexp + 1, n_reps, sampling_method.name, target_dist.name, ndims, len(samples_acc)/1000.0,
                (len(samples_acc)/max_samples)*100, h, m, s), end="\r", flush=True)

            t_ini = time.time()

            samples_acc, samples_weights_acc = sampling_method.importance_sample(target_d=target_dist,
                                                                                 n_samples=n_samples,
                                                                                 timeout=max_sampling_time - sampling_time)

            samples_logprob_acc = sampling_method.log_prob(samples_acc)

            n_samples = len(samples_acc) + batch_samples

            sampling_time += time.time()-t_ini

            sampling_method._update_model()

            if filename is not None:
                # t_ini = time.time()
                [js_div, bhattacharyya_dist, ev_mse] = \
                    evaluate_proposal(sampling_method, target_dist, space_min, space_max, sampling_eval_samples)
                # print("Evaluation time: %5.3f nsamples: %d samples/sec: %5.3f" % (time.time() - t_ini, len(samples_acc), len(samples_acc) / (time.time() - t_ini)))

                log_print("%02d %04d %7.5f %7.5f %8.5f %7.5f %7.4f %s %s %5.3f %d %d %d" % (
                    ndims, len(samples_acc), js_div, bhattacharyya_dist, ev_mse, sampling_method.get_NESS(), sampling_time,
                    sampling_method.name, target_dist.name, sampling_method.get_acceptance_rate(),
                    sampling_method.num_proposal_samples, sampling_method.num_proposal_evals, sampling_method.num_target_evals), file=filename)

                if debug:
                    plt.suptitle("%s | #smpl: %d | JSD: %3.3f | BHT: %3.3f | NESS: %.3f | AcceptRatio:%3.3f" %
                                 (sampling_method.name, n_samples, js_div, bhattacharyya_dist, sampling_method.get_NESS(),
                                  sampling_method.get_acceptance_rate()))

            if sampling_time > max_sampling_time:
                break

            # DEBUG CODE HERE
            if debug:
                # Remove previous points
                for element in pts:
                    element.remove()
                pts.clear()
                if ndims == 1:
                    pts.extend(sampling_method.draw(ax))
                    # pts.extend(ax.plot(samples_acc, samples_logprob_acc, "r."))
                    pts.extend(ax.plot(samples_acc, np.ones_like(samples_logprob_acc)*0.1, "r|", label="samples"))
                    plt.pause(0.01)
                    if videofile is not None:
                        vid_writer.add_frame(fig)
                if ndims == 2:
                    pts.extend(sampling_method.draw(ax))
                    pts.append(ax.scatter(samples_acc[:, 0], samples_acc[:, 1], label="samples", c="r", marker=".", alpha=0.4))
                    if videofile is not None:
                        vid_writer.add_frame(fig)
                    plt.pause(0.01)
                plt.legend(framealpha=0.5, loc="best")

        h, m, s = time_to_hms(time.time() - t_start)
        print(
            "Exp %02d/%02d || %s || dist: %s || dims: %d || #samples: %.1fk || %5.1f%% || t: %02dh %02dm %4.1fs" % (
                nexp + 1, n_reps, sampling_method.name, target_dist.name, ndims, len(samples_acc) / 1000.0,
                (len(samples_acc) / max_samples) * 100, h, m, s), end="\n", flush=True)

        if debug and (ndims == 1 or ndims == 2):
            if videofile is not None:
                vid_writer.add_frame(fig)
                vid_writer.save()

    if debug and (ndims == 1 or ndims == 2):
        plt.close()

    return sampling_method.get_viz_frames()
