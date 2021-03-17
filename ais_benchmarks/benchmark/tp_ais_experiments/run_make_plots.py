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
import matplotlib
import matplotlib.pyplot as plt
import yaml
from ais_benchmarks.benchmark.CBenchmark import CBenchmark
matplotlib.style.use('default')


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
    metrics = yaml.load(f, Loader=yaml.SafeLoader)["metrics"]
print("Using %d metrics: " % len(metrics), metrics)

# Read results file
data = pd.read_table(r_file, sep=" ", index_col=False, skipinitialspace=True)

# Obtain methods within the results file
method_names = data["method"].unique()
print("Found %d methods: " % len(method_names), method_names)
method_names = ["TP_AISr_ESS95", "m_pmc", "mcmc_mh", "dm_pmc",  "lais",   "multi_nested", "HiDaisee_90"]
method_labels = ["TP-AIS (ours)", "M-PMC[2]", "MCMC-MH[8]", "DM-PMC[1]", "LAIS[3]", "multi-nest[27]", "HiDaisee[4]"]

# Obtain target dists within the results file
targets = data[["target_d", "dims"]].drop_duplicates()
print("Found %d targets: " % len(targets["target_d"]), targets["target_d"].to_list())
print("with dimensions : ", targets["dims"].to_list())

legend_ready = False

metric_labels = {"MEM": "Memory (MB)",
                 "T": "Time (s)",
                 "KLD": "Kullback-Leibler Divergence",
                 "JSD": "Jensen-Shannon Divergence",
                 "EVMSE": "Expected Value Mean Squared Error"}

for metric in metrics:
    for dims, dist in zip(targets["dims"].to_list(), targets["target_d"].to_list()):
        # Check if data is valid
        indices_t = data["target_d"] == dist
        indices_d = data["dims"] == dims
        vals = data[(indices_t & indices_d)]

        # Get data limits filtering inf values
        ymin = vals[metric].min()
        ymax = vals[metric].replace([np.inf, -np.inf], np.nan).max()

        # TODO: Paper hack. Remove MCMC from memory plots
        if metric == "MEM":
            method_names = ["TP_AISr_ESS95", "m_pmc", "m_pmc", "dm_pmc", "lais", "HiDaisee_90"]
            method_labels = ["TP-AIS (ours)", "M-PMC[2]", "MCMC-MH[8]", "DM-PMC[1]", "LAIS[3]", "HiDaisee[4]"]
        else:
            method_names = ["TP_AISr_ESS95", "m_pmc", "mcmc_mh", "dm_pmc", "lais", "HiDaisee_90"]
            method_labels = ["TP-AIS (ours)", "M-PMC[2]", "MCMC-MH[8]", "DM-PMC[1]", "LAIS[3]", "HiDaisee[4]"]

        CBenchmark.make_2d_plot(data, "output_samples", metric, method_names, labels=method_labels,
                                selector=["dims", "target_d"], selector_val=[dims, dist])
        # plt.legend(mode="none", loc=(1.01, 0), ncol=len(method_names), prop={'size': 10}, numpoints=1)
        plt.gca().set_title("Target distribution: %dD %s" % (dims, dist))
        plt.gca().set_ylabel(metric_labels[metric])
        plt.gca().set_xlabel("# samples")
        if metric != "MEM" and metric != "JSD":
            plt.yscale("log",  nonposy='clip')
        else:
            plt.yscale("linear")
        _, ymax = plt.gca().get_ylim()

        # TODO: Paper hack. Tune x and y limits to the results available
        plt.gca().set_xlim(0, min(max(2000*dims, 10**dims), 100000))
        plt.gca().set_ylim(max(ymin, 1e-6), ymax)
        plt.savefig(plots_subdir + "%dD_%s_%s.pdf" % (dims, dist, metric), bbox_inches='tight', dpi=700)

        if not legend_ready:
            legend = plt.legend(mode="none", loc=(1.01, 0), ncol=len(method_names), prop={'size': 10}, numpoints=1)
            export_legend(legend, plots_subdir + "legend.pdf")
            legend_ready = True

        plt.close()
        print("Generated " + plots_subdir + "%dD_%s_%s.pdf" % (dims, dist, metric))
