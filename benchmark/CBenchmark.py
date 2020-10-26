import yaml
import numpy as np
import pandas as pd
from pprint import pformat
import matplotlib.pyplot as plt
import time

import distributions
import sampling_methods
from benchmark.evaluation import evaluate_method
from benchmark.plot_results import make_2d_plot
from utils.misc import time_to_hms


class CBenchmark(object):
    def __init__(self):
        # Info about the targets
        self.targets = []       # Target distributions to use for benchmark
        self.ndims = []         # Number of dimensions of each target

        # Info about the methods
        self.methods = []       # Methods to evaluate
        self.batch_sizes = []   # number of samples to generate per iteration

        # Info about the evaluation and metrics
        self.nsamples = []       # Number of samples to obtain after concluding evaluation of each method
        self.timeout = 18000     # Max time allowed for evaluation of each pair (method, target)
        self.eval_sampl = []     # Number of samples used for computing evaluation metrics
        self.metrics = []        # List of metrics to compute. Must be in the implemented metrics list
        self.n_experiments = 10  # Number of times a method is evaluated on a target
        self.rseed = 0           # Random seed in use

        # Info about result storage and representation
        self.output_file = "results.txt"            # Filename to store the text results
        self.generate_plots = False                 # Flag to enable result plot generation
        self.generate_plots_path = "results_plot/"  # Path for the generated result plots. Will generate a .png per combination of (method, target, metric)
        self.plot_dpi = 1600                        # Default dpi resolution for generating plots.
        self.display = False                        # Flag to display the current state of sampling. Useful for debug and video generation
        self.display_path = "results_figures/"      # Path to store each individual frame of the debug display

    def load_methods(self, methods_file, space_min, space_max, dims):
        self.methods.clear()
        m_yaml = open(methods_file, mode="r")
        # methods = yaml.load(m_yaml, Loader=yaml.FullLoader)
        methods = yaml.load(m_yaml, Loader=yaml.SafeLoader)
        for method in methods["methods"]:

            params = ['"space_min":np.array(%s)' % np.array2string(space_min, separator=', '), ",",
                      '"space_max":np.array(%s)' % np.array2string(space_max, separator=', '), ",",
                      '"dims":%d' % dims, ","]

            for p in method["params"].items():
                params.append('"%s":%s' % (p[0], p[1]))
                params.append(",")

            params_str = ""
            for p in params[0:len(params)-1]:
                params_str += p

            method_code = "%s(params={%s})" % (method["type"], params_str)
            m = eval(method_code)
            m.name = method["name"]
            self.methods.append(m)

    def load_config(self, config_file):
        b_yaml = open(config_file, mode="r")
        bench = yaml.load(b_yaml, Loader=yaml.SafeLoader)
        # Get the metrics to compute
        self.metrics = bench["metrics"]

        # Collect display configuration
        self.display = bench["display"]["value"]
        self.display_path = bench["display"]["display_path"]

        self.n_experiments = bench["nreps"]
        self.rseed = bench["rseed"]
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
            target_dist.domain_min = eval(target["domain_min"])
            target_dist.domain_max = eval(target["domain_max"])

            self.targets.append(target_dist)
            self.ndims.append(target_dist.dims)

    def run(self, benchmark_file, methods_file, config_file, out_file):
        self.output_file = out_file
        self.load_config(config_file)
        self.load_benchmark(benchmark_file)

        assert len(self.targets) > 0
        assert len(self.targets) == len(self.ndims) == len(self.nsamples)

        # TODO: Generate latex result tables
        # TODO: Generate the animation

        cols = "dims output_samples JSD BD ev_mse NESS time method target_d accept_rate proposal_samples proposal_evals target_evals\n"
        with open(self.output_file, 'w') as f:
            f.write(cols)

        # TODO: Check destination paths

        t_start = time.time()
        for target_dist, ndims, max_samples_dim, eval_sampl, batch_size in \
                zip(self.targets, self.ndims, self.nsamples, self.eval_sampl, self.batch_sizes):
            self.load_methods(methods_file, target_dist.domain_min, target_dist.domain_max, ndims)

            for sampling_method in self.methods:
                print("EVALUATING: %s || dims: %d || max samples: %d || target_d: %s || batch: %d" % (
                    sampling_method.name, ndims, max_samples_dim, target_dist.name, batch_size))

                sampling_method.reset()
                t_ini = time.time()
                viz_elems = evaluate_method(ndims=ndims,
                                            support=[target_dist.domain_min, target_dist.domain_max],
                                            target_dist=target_dist,
                                            sampling_method=sampling_method,
                                            max_samples=max_samples_dim,
                                            max_sampling_time=self.timeout,
                                            batch_size=batch_size,
                                            debug=self.display,
                                            metrics=self.metrics,
                                            rseed=self.rseed,
                                            n_reps=self.n_experiments,
                                            sampling_eval_samples=eval_sampl,
                                            filename=self.output_file)
                print("TOOK: %dh %dm %4.1fs" % time_to_hms(time.time()-t_ini))

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
                    print("VIZ : %dh %dm %4.1fs" % time_to_hms(time.time()-t_ini))

        print("BENCHMARK TOOK: %dh %dm %4.1fs" % time_to_hms(time.time()-t_start))

        # Make metric-wise plots for each target distribution with one serie for each evaluated method
        if self.generate_plots:
            self.make_plots()

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
                make_2d_plot(data, "output_samples", metric, methods,
                             selector=["dims", "target_d"], selector_val=[dims, dist])
                plt.gca().set_title("Target distribution: %dD %s" % (dims, dist))
                plt.gca().set_ylabel(metric)
                plt.gca().set_xlabel("# samples")
                # plt.yscale("log",  nonposy='clip')
                ymin, ymax = plt.gca().get_ylim()
                plt.gca().set_ylim(ymin, ymax * 1.2)
                plt.savefig(self.generate_plots_path + "%dD_%s_%s.pdf" % (dims, dist, metric), bbox_inches='tight', dpi=self.plot_dpi)
                plt.close()
                print("Generated " + self.generate_plots_path + "%dD_%s_%s.pdf" % (dims, dist, metric))
        print("PLOT GENERATION TOOK: %5.3fs" % (time.time()-t_start))
