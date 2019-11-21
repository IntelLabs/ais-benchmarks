import yaml
import numpy as np
import distributions
import sampling_methods

from sampling_methods.evaluation import evaluate_method


class CBenchmark(object):
    def __init__(self):
        # Info about the targets
        self.targets = []       # Target distributions to use for benchmark
        self.ndims = []         # Number of dimensions of each target
        self.space_size = []    # Min max limits of the sampled space

        # Info about the methods
        self.methods = []       # Methods to evaluate
        self.batch_sizes = []   # number of samples to generate per iteration

        # Info about the evaluation and metrics
        self.nsamples = []      # Number of samples to obtain after concluding evaluation of each method
        self.timeout = 3600     # Max time allowed for evaluation of each pair (method, target)
        self.eval_sampl = []    # Number of samples used for computing evaluation metrics
        self.metrics = []       # List of metrics to compute. Must be in the implemented metrics list

        # Info about result storage and representation
        self.output_file = "results.txt"            # Filename to store the text results
        self.generate_plots = False                 # Flag to enable result plot generation
        self.generate_plots_path = "results_plot/"  # Path for the generated result plots. Will generate a .png per combination of (method, target, metric)
        self.display = False                        # Flag to display the current state of sampling. Useful for debug and video generation
        self.display_path = "results_figures/"      # Path to store each individual frame of the debug display

    def load_methods(self, methods_file, space_min, space_max, dims):
        self.methods.clear()
        m_yaml = open(methods_file, mode="r")
        methods = yaml.load(m_yaml)
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

    def load_benchmark(self, benchmark_file):
        # Clear previously loaded benchmark configuration
        self.targets.clear()
        self.ndims.clear()
        self.space_size.clear()
        self.batch_sizes.clear()
        self.eval_sampl.clear()
        self.nsamples.clear()

        b_yaml = open(benchmark_file, mode="r")
        bench = yaml.load(b_yaml)

        # Get the metrics to compute
        self.metrics = bench["metrics"]

        # Collect display configuration
        self.display = bench["display"]["value"]
        self.display_path = bench["display"]["display_path"]

        # Collect output configuration
        self.output_file = bench["output"]["file"]
        self.generate_plots = bench["output"]["make_plots"]
        self.generate_plots_path = bench["output"]["plots_path"]

        for target in bench["targets"]:
            # Collect the target specific evaluation parameters
            self.nsamples.append(target["nsamples"])
            self.eval_sampl.append(target["nsamples_eval"])
            self.batch_sizes.append(target["batch_size"])

            # Build the target distribution
            params = []
            for p in target["params"].items():
                params.append("%s=np.array(%s)" % (p[0], p[1]))
                params.append(",")

            params_str = ""
            for p in params[0:len(params)-1]:
                params_str += p
            dist_code = "%s(%s)" % (target["type"], params_str)
            target_dist = eval(dist_code)
            if target_dist is None:
                raise ValueError("Error creating target dist: %s" % dist_code)
            target_dist.name = target["name"]
            target_dist.domain_min = eval(target["domain_min"])
            target_dist.domain_max = eval(target["domain_max"])

            self.targets.append(target_dist)
            self.ndims.append(target_dist.dims)
            self.space_size.append(np.array(target["space_size"]))

    def run(self, benchmark_file, methods_file):

        self.load_benchmark(benchmark_file)

        assert len(self.targets) > 0
        assert len(self.targets) == len(self.ndims) == len(self.space_size) == len(self.nsamples)

        # TODO: Use the batch size
        # TODO: Use the display plot paths
        # TODO: Use the metrics

        for target_dist, ndims, space_size, max_samples_dim, eval_sampl in zip(self.targets, self.ndims, self.space_size, self.nsamples, self.eval_sampl):
            self.load_methods(methods_file, target_dist.domain_min, target_dist.domain_max, ndims)

            for sampling_method in self.methods:
                print("EVALUATING: %s with %d max samples %d dims" % (sampling_method.name, max_samples_dim, ndims))
                sampling_method.reset()
                evaluate_method(ndims, space_size, target_dist, sampling_method, max_samples_dim,
                                eval_sampl, max_sampling_time=self.timeout,
                                debug=self.display, filename=self.output_file)
