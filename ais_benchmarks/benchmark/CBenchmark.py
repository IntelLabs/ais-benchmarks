import yaml
import numpy as np
import pandas as pd
from pprint import pformat
import matplotlib.pyplot as plt
import time
import cProfile
import pstats
import pathlib
import os
import sys

import ais_benchmarks.distributions
import ais_benchmarks.sampling_methods
from ais_benchmarks.sampling_methods.base import t_tensor
from ais_benchmarks.metrics.divergences import CKLDivergence
from ais_benchmarks.metrics.divergences import CJSDivergence
from ais_benchmarks.metrics.performance import CMemoryUsage
from ais_benchmarks.metrics.performance import CElapsedTime
from ais_benchmarks.metrics.statistics import CExpectedValueMSE
from ais_benchmarks.utils.misc import time_to_hms
from ais_benchmarks.utils.plot_utils import plot_pdf
from ais_benchmarks.utils.plot_utils import grid_sample_distribution
from ais_benchmarks.utils.plot_utils import plot_grid_sampled_pdfs


class CBenchmark(object):
    def __init__(self):
        # Info about the targets
        self.targets = []  # Target distributions to use for benchmark
        self.ndims = []  # Number of dimensions of each target

        # Info about the methods
        self.methods = []  # Methods to evaluate
        self.batch_sizes = []  # number of samples to generate per iteration

        # Info about the evaluation and metrics
        self.nsamples = []  # Number of samples to obtain after concluding evaluation of each method
        self.timeout = 18000  # Max time allowed for evaluation of each pair (method, target)
        self.eval_sampl = []  # Number of samples used for computing evaluation metrics
        self.metrics = []  # List of metrics to compute. Must be in the implemented metrics list
        self.n_experiments = 10  # Number of times a method is evaluated on a target
        self.rseed = 0  # Random seed in use

        # Info about result storage and representation
        self.output_file = "results.txt"  # Filename to store the text results
        self.generate_plots = False  # Flag to enable result plot generation
        self.generate_plots_path = "results_plot/"  # Path for the generated result plots. Will generate a .png per combination of (method, target, metric)
        self.plot_dpi = 1600  # Default dpi resolution for generating plots.
        self.debug_text = True  # Flag to display sampling progress on the standard output.
        self.debug_plot = False  # Flag to show a plot with the sampling progress. VERY SLOW!
        self.debug_plot_save = False  # Flag to save each debug plot figure the current state of sampling.
        self.debug_plot_save_path = "results_figures/"  # Path to store each individual frame of the debug display

    def load_methods(self, methods_file, space_min, space_max, dims):
        self.methods.clear()
        m_yaml = open(methods_file, mode="r")
        methods = yaml.load(m_yaml, Loader=yaml.SafeLoader)
        for method in methods["methods"]:

            params = ['"space_min":np.array(%s)' % np.array2string(space_min, separator=', '), ",",
                      '"space_max":np.array(%s)' % np.array2string(space_max, separator=', '), ",",
                      '"dims":%d' % dims, ","]

            for p in method["params"].items():
                params.append('"%s":%s' % (p[0], p[1]))
                params.append(",")

            params_str = ""
            for p in params[0:len(params) - 1]:
                params_str += p

            method_code = "%s(params={%s})" % (method["type"], params_str)
            m = eval(method_code)
            m.name = method["name"]
            m.debug = method["debug"]
            self.methods.append(m)

    def load_config(self, config_file):
        b_yaml = open(config_file, mode="r")
        bench = yaml.load(b_yaml, Loader=yaml.SafeLoader)
        # Get the metrics to compute
        self.metrics = bench["metrics"]

        # TODO: look at the config file and adapt the configuration with the benchmarking configuration, change
        #       variable names accordingly to their purpose. Refactor the debug mode passed to the algos
        #       and review that it is used by the algo to display useful debug info, text and plots.
        # Collect display configuration

        # Collect debug configuration
        self.debug_text = bench["debug"]["text"]
        self.debug_plot = bench["debug"]["plot"]["show"]
        self.debug_plot_save = bench["debug"]["plot"]["save"]
        self.debug_plot_save_path = bench["debug"]["plot"]["path"]

        # Experiment configuration
        self.n_experiments = bench["nreps"]
        self.rseed = bench["rseed"]
        if self.rseed >= 0:
            np.random.seed(self.rseed)

        # Collect output configuration
        # self.output_file = bench["output"]["file"]
        self.generate_plots = bench["output"]["make_plots"]
        self.generate_plots_path = bench["output"]["plots_path"]
        self.plot_dpi = bench["output"]["plots_dpi"]

    def load_benchmark(self, benchmark_file):
        # Clear previously loaded benchmark configuration
        self.targets.clear()
        self.ndims.clear()
        self.batch_sizes.clear()
        self.eval_sampl.clear()
        self.nsamples.clear()

        b_yaml = open(benchmark_file, mode="r")
        bench = yaml.load(b_yaml, Loader=yaml.SafeLoader)

        for target in bench["targets"]:
            # Collect the target specific evaluation parameters
            self.nsamples.append(target["nsamples"])
            self.eval_sampl.append(target["nsamples_eval"])
            self.batch_sizes.append(target["batch_size"])

            # Build the target distribution
            dist_code = "%s(%s)" % (target["type"], pformat(target["params"]))
            try:
                target_dist = eval(dist_code)
            except BaseException as e:
                print(e)
                raise ValueError("Error creating target dist: %s" % dist_code)
            target_dist.name = target["name"]
            self.targets.append(target_dist)
            self.ndims.append(target_dist.dims)

    @staticmethod
    def _check_make_paths(path):
        if not pathlib.Path(path).exists():
            print("WARNING. Debug plot save path does not exist. Creating it...")
            try:
                os.makedirs(path)
            except OSError as ex:
                print("ERROR: Failed to create directory: %s. %s" %
                      (path, str(ex)), file=sys.stderr)
            else:
                print("Created directory: %s." % path)

    def run(self, benchmark_file, methods_file, config_file, out_file):
        self.output_file = out_file
        self.load_config(config_file)
        self.load_benchmark(benchmark_file)

        assert len(self.targets) > 0
        assert len(self.targets) == len(self.ndims) == len(self.nsamples)

        # Check destination paths and create if not existing
        CBenchmark._check_make_paths(self.debug_plot_save_path)
        CBenchmark._check_make_paths(self.generate_plots_path)
        CBenchmark._check_make_paths(pathlib.Path(self.output_file).parent)

        # Enable / disable debug plot save path
        self.debug_plot_save_path = None if not self.debug_plot_save else self.debug_plot_save_path

        cols = "dims output_samples " + " ".join([m for m in self.metrics]) + \
               " NESS method target_d accept_rate proposal_samples proposal_evals target_evals\n"
        with open(self.output_file, 'w') as f:
            f.write(cols)

        t_start = time.time()
        for target_dist, ndims, max_samples_dim, eval_sampl, batch_size in \
                zip(self.targets, self.ndims, self.nsamples, self.eval_sampl, self.batch_sizes):
            self.load_methods(methods_file,
                              target_dist.support()[0],
                              target_dist.support()[1], ndims)

            for sampling_method in self.methods:
                print("EVALUATING: %s || dims: %d || max samples: %d || target_d: %s || batch: %d" % (
                    sampling_method.name, ndims, max_samples_dim, target_dist.name, batch_size))

                sampling_method.reset()
                t_ini = time.time()
                viz_elems = CBenchmark.evaluate_method(ndims=ndims,
                                                       target_dist=target_dist,
                                                       sampling_method=sampling_method,
                                                       max_samples=max_samples_dim,
                                                       max_sampling_time=self.timeout,
                                                       batch_size=batch_size,
                                                       debug=self.debug_plot,
                                                       metrics=self.metrics,
                                                       rseed=self.rseed,
                                                       n_reps=self.n_experiments,
                                                       sampling_eval_samples=eval_sampl,
                                                       filename=self.output_file,
                                                       debug_plot_path=self.debug_plot_save_path)
                print("TOOK: %dh %dm %4.1fs" % time_to_hms(time.time() - t_ini))

                if viz_elems is not None:
                    t_ini = time.time()
                    import visualization.visuals as viz
                    from visualization.matplotlib.viz_interface import draw_frames
                    fig = plt.figure(figsize=(13, 15))
                    plt.axis('off')
                    plt.show(block=False)

                    x_axis = viz.CAxis(id=-1,
                                       start=np.array([target_dist.domain_min, 0, 0]),
                                       end=np.array([target_dist.domain_max, 0, 0]))
                    y_axis = viz.CAxis(id=-2, end=np.array([0, 1, 0]))
                    y_axis.ticks_size = [.1] * len(y_axis.ticks)
                    x_axis.ticks_size = [.01] * len(x_axis.ticks)

                    target_d_viz = viz.CFunction(id=-3,
                                                 limits=[target_dist.domain_min, target_dist.domain_max],
                                                 func=target_dist.prob,
                                                 resolution=1000)

                    target_d_viz.outline_color = viz.CColor.BLUE

                    print("Drawing %d visual elements" % len(viz_elems))
                    draw_frames(frames=viz_elems, static_elems=[x_axis, y_axis, target_d_viz])
                    print("VIZ : %dh %dm %4.1fs" % time_to_hms(time.time() - t_ini))

        print("BENCHMARK TOOK: %dh %dm %4.1fs" % time_to_hms(time.time() - t_start))

        # Make metric-wise plots for each target distribution with one serie for each evaluated method
        if self.generate_plots:
            self.make_plots()

        # TODO: Generate latex result tables
        # TODO: Generate the animation

    def make_plots(self, benchmark_file=None, methods_file=None, config_file=None):
        if benchmark_file is not None:
            self.load_benchmark(benchmark_file)

        if config_file is not None:
            self.load_config(config_file)

        if methods_file is not None:
            for target_dist, ndims in zip(self.targets, self.ndims):
                self.load_methods(methods_file, target_dist.domain_min, target_dist.domain_max, ndims)

        t_start = time.time()
        for target_d in self.targets:
            methods = [m.name for m in self.methods]  # for all evaluated methods
            data = pd.read_table(self.output_file, sep=" ", index_col=False, skipinitialspace=True)
            for metric in self.metrics:
                # TODO: Check that data exists or throw an error otherwise
                [dist, dims] = [target_d.name, target_d.dims]
                CBenchmark.make_2d_plot(data, "output_samples", metric, methods,
                                        selector=["dims", "target_d"], selector_val=[dims, dist])
                plt.gca().set_title("Target distribution: %dD %s" % (dims, dist))
                plt.gca().set_ylabel(metric)
                plt.gca().set_xlabel("# samples")
                # plt.yscale("log",  nonposy='clip')
                ymin, ymax = plt.gca().get_ylim()
                plt.gca().set_ylim(ymin, ymax * 1.2)
                plt.savefig(self.generate_plots_path + "%dD_%s_%s.pdf" % (dims, dist, metric),
                            bbox_inches='tight', dpi=self.plot_dpi)
                plt.close()
                print("Generated " + self.generate_plots_path + "%dD_%s_%s.pdf" % (dims, dist, metric))
        print("PLOT GENERATION TOOK: %5.3fs" % (time.time() - t_start))

    @staticmethod
    def make_2d_plot(data, xaxis, yaxis, methods, selector=None, selector_val=None, labels=None, mark_points=10):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        markers = ["x", "d", "+", "*", 4, 5, 6, 7]

        for m_id, m in enumerate(methods):
            m_lbl = labels[m_id] if labels is not None else m

            indices = data['method'] == m
            for sel, val in zip(selector, selector_val):
                indices = indices & (data[sel] == val)

            x_data = data[indices].groupby(xaxis)[xaxis].mean()
            y_data = data[indices].groupby(xaxis)[yaxis]
            if len(y_data) == 0:
                continue
            y_err = np.sqrt(y_data.mean().rolling(window=10).std())
            # y_err = np.sqrt(y_data.std())
            y_avg = y_data.mean().rolling(window=10).mean()
            markevery = max(1, int(len(x_data.values) / mark_points))
            p = ax.plot(x_data.values, y_avg.values, label=m_lbl, marker=markers[m_id % len(markers)],
                        markevery=markevery)
            color = p[-1].get_color()
            plt.fill_between(x_data.values, y_avg - y_err, y_avg + y_err, alpha=0.1, edgecolor=color, facecolor=color)

        ax.set_xlabel(xaxis)
        ax.set_ylabel(yaxis)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(0, ymax * 1.1)
        # ax.legend(mode="expand", loc=(1,0), ncol=1, prop={'size': 10}, numpoints=1)
        # ax.legend(mode="none", loc=(1,0), ncol=1, prop={'size': 10}, numpoints=1)

    @staticmethod
    def write_results(results, method, target, nsamples, file):
        with open(file, mode='a+') as f:
            text = "%02d %04d " % (target.dims, nsamples)
            text += " ".join(["%9.5f" % val for val in results.values()])
            text += "%9.3f %s %s %5.3f %d %d %d" % (method.get_NESS(), method.name, target.name,
                                                    method.get_acceptance_rate(), method.num_proposal_samples,
                                                    method.num_proposal_evals, method.num_target_evals)
            f.write(text + "\n")

    @staticmethod
    def evaluate_method(ndims, target_dist, sampling_method, max_samples, sampling_eval_samples,
                        metrics=("NESS", "JSD", "T"), rseed=-1, n_reps=10, batch_size=16,
                        debug=True, filename=None, max_sampling_time=600, profile=False, debug_plot_path=None):
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
            Random seed to be used. If rseed < 0, the random generator state is not altered. Default: -1

        n_reps: int
            Number of times to repeat each experiment. Default: 10.

        batch_size: int
            Number of samples to be generated before computing the partial metrics. Default: 16.

        debug: bool
            Flag to enable console debug messages and other debugging visualizations. Default: False.

        debug_plot_path: str
            Path to store each frame of the debug visualizations. Use None to prevent saving the debug plot.
            Default: None.

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
        if rseed >= 0:
            np.random.seed(rseed)

        # Start profiling tools
        profiler = None
        if profile:
            profiler = cProfile.Profile()
            profiler.enable()

        # Create metrics instances
        metrics_eval = list()
        for m in metrics:
            if m == "KLD":
                metrics_eval.append(CKLDivergence())
            elif m == "JSD":
                metrics_eval.append(CJSDivergence())
            elif m == "NESS":
                pass
            elif m == "EVMSE":
                metrics_eval.append(CExpectedValueMSE())
            elif m == "T":
                metrics_eval.append(CElapsedTime())
            elif m == "MEM":
                metrics_eval.append(CMemoryUsage())

        # TODO: Cleanup the debug viz code
        # Initialize visualization
        if debug:
            pts = []
            if ndims == 1:
                fig = plt.figure(figsize=(10, 8))
                ax = plt.subplot(111)
                plt.show(block=False)
                plot_pdf(ax, target_dist, np.array(target_dist.support()[0]), np.array(target_dist.support()[1]),
                         alpha=1.0, options="b-", resolution=0.01, label="$\pi(x)$")

                plt.xlim(np.array(target_dist.support()[0]), np.array(target_dist.support()[1]))
                plt.ylim(0, ax.get_ylim()[1])

            elif ndims == 2:
                fig = plt.figure(figsize=(10, 8))
                ax = plt.subplot(111)
                plt.show(block=False)

                grid, log_prob, dims, shape = grid_sample_distribution(target_dist,
                                                                       np.array(target_dist.support()[0]),
                                                                       np.array(target_dist.support()[1]),
                                                                       resolution=0.02)
                plot_grid_sampled_pdfs(ax, dims, np.exp(log_prob), shape=shape, alpha=1, label="$\pi(x)$", cmap='gray',
                                       linestyles='dashed')

                plt.xlim(np.array(target_dist.support()[0][0]), np.array(target_dist.support()[1][0]))
                plt.ylim(np.array(target_dist.support()[0][1]), np.array(target_dist.support()[1][1]))

            elif ndims == 3:
                fig = plt.figure(figsize=(10, 8))
                ax = plt.subplot(111, projection='3d')
                plt.show(block=False)

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
            niter = 0
            while len(samples_acc) < max_samples:

                # Obtain experiment execution runtime
                hr, mn, sec = time_to_hms(time.time() - t_start)

                # Display partial sampling experiment statistics
                text_display = "%02d/%02d | %12s | %6s | %dD | #s: %.1fk | %5.1f%% | t: %02dh %02dm %4.1fs | " % (
                    nexp + 1, n_reps, sampling_method.name, target_dist.name, ndims, len(samples_acc) / 1000.0,
                    (len(samples_acc) / max_samples) * 100, hr, mn, sec)

                text_display += " | ".join(["%s: %7.5f" % (m.name, m.value) for m in metrics_eval])
                # print(text_display, end="\r", flush=True)
                print(text_display)

                # Initialize metrics before running the evaluated code
                [m.pre() for m in metrics_eval]

                # Perform importance sampling operation to generate a batch of samples and compute the time taken
                samples_acc, _ = sampling_method.importance_sample(target_d=target_dist, n_samples=n_samples,
                                                                   timeout=max_sampling_time - sampling_time)

                # Compute metrics right after the sampling operation
                [m.post() for m in metrics_eval]

                # Sampling methods generate a desired number of samples and maintain the state. Therefore it is not
                # possible to ask the sampling algos for batch_size samples every time, thus in order to have partial
                # results during the sampling experiments, the number of samples to generate is always increase by
                # batch size. This lets us measure partial results in increments of batch_size samples.
                n_samples = len(samples_acc) + batch_size

                # TODO: Cleanup the debug viz code
                # Display visualization of sampling procedure
                if debug:
                    # plt.title(text_display)
                    plt.title(sampling_method.name)
                    # Remove previous points
                    for element in pts:
                        element.remove()
                    pts.clear()
                    if ndims == 1:
                        pts.extend(sampling_method.draw(ax))
                        # Display samples
                        pts.extend(ax.plot(samples_acc, np.ones(len(samples_acc)) * 0.05, "r|", label="samples"))
                        plt.pause(0.01)
                    if ndims == 2:
                        pts.extend(sampling_method.draw(ax))
                        pts.append(ax.scatter(samples_acc[:, 0], samples_acc[:, 1], label="samples", c="r", marker=".",
                                              alpha=0.4))
                        plt.pause(0.01)
                    if ndims == 3:
                        pts.extend(sampling_method.draw(ax))
                        pts.append(ax.scatter(samples_acc[:, 0], samples_acc[:, 1], samples_acc[:, 2], label="samples",
                                              c="r", marker=".", alpha=0.4))
                        plt.pause(0.01)
                    if ndims <= 3:
                        plt.legend(framealpha=0.5, loc="best")
                        if debug_plot_path is not None:
                            print("\nSave fig: " + debug_plot_path + "%s_%s_%dD_run%d_it%d.png" % (
                            sampling_method.name, target_dist.name, target_dist.dims, nexp, niter))
                            plt.savefig(debug_plot_path + "%s_%s_%dD_%d_run%d_it.png" %
                                        (sampling_method.name, target_dist.name, target_dist.dims, nexp, niter),
                                        bbox_inches='tight')

                # Compute metrics
                results = dict()
                for m in metrics_eval:
                    val = m.compute(p=target_dist, q=sampling_method, nsamples=sampling_eval_samples)
                    results[m.name] = val

                # Write metric results to file
                if filename is not None:
                    CBenchmark.write_results(results=results, method=sampling_method, target=target_dist,
                                             nsamples=len(samples_acc), file=filename)

                # Some sampling algorithms may take too long to generate the desired number of samples, the
                # timeout ensures the operation will end if the specified timeout is reached.
                if sampling_time > max_sampling_time:
                    break

                niter += 1

            # Obtain experiment execution runtime
            h, m, s = time_to_hms(time.time() - t_start)

            # Print final experiment statistics
            text_display = "%02d/%02d | %12s | %6s | %dD | #s: %.1fk | %5.1f%% | t: %02dh %02dm %4.1fs | " % (
                nexp + 1, n_reps, sampling_method.name, target_dist.name, ndims, len(samples_acc) / 1000.0,
                (len(samples_acc) / max_samples) * 100, hr, mn, sec)

            text_display += " | ".join(["%s: %7.5f" % (m.name, m.value) for m in metrics_eval])
            print(text_display, end="\n", flush=True)

        # Stop profiling tools and save results
        if profiler is not None:
            profiler.disable()
            ps = pstats.Stats(profiler)
            ps.sort_stats("cumtime")
            ps.dump_stats(filename + ".profile")
            ps.print_stats()

        if debug:
            plt.close()

        # Return sampling visualization data
        return sampling_method.get_viz_frames()
