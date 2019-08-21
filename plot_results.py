import os
import fnmatch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def load_files_regexp(path, regexp):
    path = "/home/jfelip/workspace/prob-comp-code/sampling_experiments/"
    filenames = fnmatch.filter(os.listdir(path), regexp)
    series = []

    for file in filenames:
        series.append(np.loadtxt(path+file))


def make_2d_plot(data, xaxis, yaxis, methods, selector=None, selector_val=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.hold(True)

    for m in methods:
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
        p = ax.plot(x_data.values, y_avg.values, label=m)
        color = p[-1].get_color()
        plt.fill_between(x_data.values, y_avg - y_err, y_avg + y_err, alpha=0.1, edgecolor=color, facecolor=color)

    ax.set_xlabel(xaxis)
    ax.set_ylabel(yaxis)
    ax.legend()


def make_2d_barplot(data, xaxis, yaxis, methods, bar_points, selector=None, selector_val=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.hold(True)

    n_series = len(methods)
    n_groups = len(bar_points)
    bar_width = 0.8 / n_series
    x_ticks = [p for p in range(len(bar_points))]

    colors = cm.get_cmap("Pastel1")
    hatches = ["","\\\\", "oo", "xxx", "---", "+", "\\", "|", "O", ".", "*"]

    for m_id, m in enumerate(methods):

        indices = data['method'] == m
        for sel, val in zip(selector, selector_val):
            indices = indices & (data[sel] == val)

        x_data = data[indices].groupby(xaxis)[xaxis].mean()
        y_data = data[indices].groupby(xaxis)[yaxis].mean()
        if len(y_data) == 0:
            continue
        y_err = np.sqrt(y_data.rolling(window=10, min_periods=1).std())
        y_avg = y_data.rolling(window=10, min_periods=1).mean()
        for xtick,bar_point in zip(x_ticks,bar_points):
            # ax.bar(xtick, y_avg[x_data.values==bar_point].values, yerr=y_err[x_data.values==bar_point].values, label=m)
            x_idx = x_data.index.get_loc(bar_point, method="nearest")
            yval = y_avg.iloc[x_idx]
            yerr = y_err.iloc[x_idx]
            ax.bar(xtick + bar_width*m_id, yval, width=bar_width, yerr=yerr, label=m if xtick==x_ticks[0] else "",
                   color=colors(m_id/9.0), ecolor="k", hatch=hatches[m_id])

    ax.set_xticks([r+(bar_width*n_series/2) for r in x_ticks])
    ax.set_xticklabels([str(b) for b in bar_points])
    ax.set_xlabel(xaxis)
    ax.set_ylabel(yaxis)
    ax.set_yscale("log")
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax*10)
    ax.legend(mode="expand", loc=9, ncol=4)


# dims samples kl_kde bhat_kde kl_nn bhat_nn time method final_samples
data = pd.read_table("results.txt", delim_whitespace=True)

dimensions = [1,2,3,4,5,6,7,8,9]
dists = ["gmm", "normal", "egg"]
path = "results/"
dpi = 1600
for dist in dists:
    for dims in dimensions:
        methods = ["TP", "MCMC-MH", "nested", "multi-nested"]
        bar_points = [5, 10, 25, 50, 100, 200, 500, 1000, 2000]
        make_2d_barplot(data, "output_samples", "time", methods, bar_points=bar_points, selector=["dims", "target_dist"], selector_val=[dims, dist])
        plt.gca().set_title("Target distribution: %s, Dimensions: %d" % (dist, dims))
        plt.gca().set_ylabel("time(s)")
        plt.gca().set_xlabel("# samples")
        plt.yscale("log",  nonposy='clip')
        plt.savefig(path + "%d_dims_%s_dist_time.pdf" % (dims, dist), bbox_inches='tight', dpi=dpi)
        # plt.show()
        plt.close()

        # make_2d_plot(data, "output_samples", "kl_nn", methods, selector=["dims", "target_dist"], selector_val=[dims, dist])
        make_2d_barplot(data, "output_samples", "kl_nn", methods, bar_points=bar_points,
                        selector=["dims", "target_dist"], selector_val=[dims, dist])
        plt.gca().set_title("Target distribution: %s, Dimensions: %d, Approximation: NN" % (dist, dims))
        plt.gca().set_ylabel("KL Divergence")
        plt.gca().set_xlabel("# samples")
        plt.yscale("log")
        plt.savefig(path + "%d_dims_%s_dist_nn.pdf" % (dims, dist), bbox_inches='tight', dpi=dpi)
        # plt.show()
        plt.close()

        # make_2d_plot(data, "output_samples", "kl_kde", methods, selector=["dims", "target_dist"], selector_val=[dims, dist])
        make_2d_barplot(data, "output_samples", "kl_kde", methods, bar_points=bar_points,
                        selector=["dims", "target_dist"], selector_val=[dims, dist])
        plt.gca().set_title("Target distribution: %s, Dimensions: %d, Approximation: KDE" % (dist, dims))
        plt.gca().set_ylabel("KL Divergence")
        plt.gca().set_xlabel("# samples")
        plt.savefig(path + "%d_dims_%s_dist_kde.pdf" % (dims, dist), bbox_inches='tight', dpi=dpi)
        # plt.show()
        plt.close()

        print("Generated " + path + "%d_dims_%s_dist" % (dims, dist))

        # make_table(data, "output_samples", "kl_kde", methods, selector=["dims", "target_dist"], selector_val=[dims, dist])
        #
        # nsamples=[] dims=[] target=[] eval_method=[]
        # method:
        # nsamples
        # 5
        # 10
        # 25
        # 50
        # 100
        # 200
        # 500
        # 1000
        # 2000
        #
        # target:
        # gmm
        # normal



# TODO: Get table at specific sample number values
# TODO: Curate the data such that there is data points in the selected sample numbers

