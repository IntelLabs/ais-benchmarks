"""
Traverse methods and make available series (methods)

Traverse benchmark definitions and compile a list of methods and dimensions

For each metric in the config file
    For each benchmark in benchmarks subdir
        Create a plot with [methods] as series
"""

import os
import pandas as pd
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import yaml
from benchmark.CBenchmark import CBenchmark
from cycler import cycler
plt.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')


def export_legend(legend, filename="legend.png"):
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


def_path = str(pathlib.Path(__file__).parent.absolute()) + os.sep
plots_subdir = def_path + "plots" + os.sep
r_file = def_path + "results.txt"
c_file = def_path + "def_config.yaml"

# Get metrics to process
with open(c_file, 'r') as f:
    metrics = yaml.load(f)["metrics"]

# Read results file
data = pd.read_table(r_file, sep=" ", index_col=False, skipinitialspace=True)

# Obtain methods within the results file
method_names = data["method"].unique()
print("Found %d methods: " % len(method_names), method_names)
method_names = ["lais", "mcmc_mh", "dm_pmc", "m_pmc", "TP_AISr_ESS90", "HiDaisee_90", "multi_nested"]
method_labels = ["lais", "mcmc_mh", "dm_pmc", "m_pmc", "TP_AISr_ESS90", "HiDaisee_90", "multi_nested"]

# Obtain target dists within the results file
targets = data[["target_d", "dims"]].drop_duplicates()
print("Found %d targets: " % len(targets["target_d"]), targets["target_d"].to_list())
print("with dimensions : ", targets["dims"].to_list())

legend_ready = False

for metric in metrics:
    for dims, dist in zip(targets["dims"].to_list(), targets["target_d"].to_list()):
        # Check if data is valid
        indices_t = data["target_d"] == dist
        indices_d = data["dims"] == dims
        vals = data[(indices_t & indices_d)]

        # Get data limits
        ymin = vals[metric].min()
        ymax = vals[metric].replace([np.inf, -np.inf], np.nan).max()

        CBenchmark.make_2d_plot(data, "output_samples", metric, method_names,
                                selector=["dims", "target_d"], selector_val=[dims, dist])
        # plt.legend(mode="none", loc=(1.01, 0), ncol=len(method_names), prop={'size': 10}, numpoints=1)
        plt.gca().set_title("Target distribution: %dD %s" % (dims, dist))
        plt.gca().set_ylabel(metric)
        plt.gca().set_xlabel("# samples")
        plt.yscale("log",  nonposy='clip')
        # ymin, ymax = plt.gca().get_ylim()
        plt.gca().set_ylim(max(ymin, 1e-6), ymax)
        plt.savefig(plots_subdir + "%dD_%s_%s.pdf" % (dims, dist, metric), bbox_inches='tight', dpi=700)

        if not legend_ready:
            legend = plt.legend(mode="none", loc=(1.01, 0), ncol=len(method_names), prop={'size': 10}, numpoints=1)
            export_legend(legend, plots_subdir + "legend.pdf")
            legend_ready = True

        plt.close()
        print("Generated " + plots_subdir + "%dD_%s_%s.pdf" % (dims, dist, metric))
