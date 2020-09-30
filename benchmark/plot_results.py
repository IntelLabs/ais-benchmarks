import os
import fnmatch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import math


def load_files_regexp(path, regexp):
    path = "/home/jfelip/workspace/prob-comp-code/sampling_experiments/"
    filenames = fnmatch.filter(os.listdir(path), regexp)
    series = []

    for file in filenames:
        series.append(np.loadtxt(path+file))


def make_2d_plot(data, xaxis, yaxis, methods, selector=None, selector_val=None, labels=None, mark_points=10):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    markers = ["x", "d", "+", "*", 4, 5, 6, 7]

    for m_id,m in enumerate(methods):
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
        p = ax.plot(x_data.values, y_avg.values, label=m_lbl, marker=markers[m_id], markevery=markevery)
        color = p[-1].get_color()
        plt.fill_between(x_data.values, y_avg - y_err, y_avg + y_err, alpha=0.1, edgecolor=color, facecolor=color)

    ax.set_xlabel(xaxis)
    ax.set_ylabel(yaxis)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(0, ymax*1.1)
    ax.legend(mode="expand", loc=9, ncol=3, prop={'size': 12}, numpoints=1)


def make_2d_barplot(data, xaxis, yaxis, methods, bar_points, selector=None, selector_val=None, labels=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.hold(True)

    n_series = len(methods)
    n_groups = len(bar_points)
    bar_width = 0.8 / n_series
    x_ticks = [p for p in range(len(bar_points))]

    colors = cm.get_cmap("Set3")
    hatches = ["","\\\\", "oo", "xx", "---", "o", "\\", "|", "O", ".", "*"]

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
            yval = yval * int((not math.isnan(yerr)))
            m_lbl = labels[m_id] if labels is not None else m
            ax.bar(xtick + bar_width * m_id, yval, width=bar_width, yerr=yerr, label=m_lbl if xtick==x_ticks[0] else "",
                   color=colors(m_id/9.0), ecolor="k", hatch=hatches[m_id])

    ax.set_xticks([r+(bar_width*n_series/2) for r in x_ticks])
    ax.set_xticklabels([str(b) for b in bar_points])
    ax.set_xlabel(xaxis)
    ax.set_ylabel(yaxis)
    # ax.set_yscale("log")
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(max(0.001, ymin), ymax*1.1)
    ax.legend(mode="expand", loc=9, ncol=3, prop={'size': 12}, numpoints=1)


if __name__ == "__main__":
    # dims samples kl_kde bhat_kde kl_nn bhat_nn time method final_samples
    data = pd.read_table("results.txt", sep=" ", index_col=False, skipinitialspace=True)

    # methods = ["TP_simple_leaf_haar", "M-PMC", "MCMC-MH", "DM_AIS", "LAIS", "multi-nested"]
    # labels = ["TP_AIS (ours)", "M-PMC[2]", "MCMC-MH[8]", "DM-PMC[26]", "LAIS[3]", "multi-nest[27]"]

    methods = ["hi_daisee", "tp_simple_leaf_haar"]
    labels = ["HiDaisee", "TP_AIS (ours)"]

    # dims output_samples JSD BD EV_MSE NESS method target_d accept_rate proposal_samples proposal_evals target_evals

    plot_mode = "lines"
    # dimensions = [1,2,3,4,5,6,7]
    dimensions = [1, 2]
    # dists = ["gmm", "normal", "egg"]
    dists = ["gmm", "normal", "banana2D"]
    path = "results" + os.sep
    dpi = 1600
    for dist in dists:
        for dims in dimensions:
            barmin = np.log(2 ** dims + 1)
            barmax = np.log(1000*dims)
            bar_points = np.logspace(barmin, barmax, num=8, base=np.e)
            bar_points = list(bar_points.astype(np.int32))

            if plot_mode == "bar":
                make_2d_barplot(data, "output_samples", "time", methods, bar_points=bar_points,
                                selector=["dims", "target_d"], selector_val=[dims, dist], labels=labels)
            else:
                make_2d_plot(data, "output_samples", "time", methods,
                             selector=["dims", "target_d"], selector_val=[dims, dist], labels=labels)
                plt.gca().set_xlim(bar_points[0], bar_points[-1])

            plt.gca().set_title("Target distribution: %s, Dimensions: %d" % (dist, dims))
            plt.gca().set_ylabel("time(s)")
            plt.gca().set_xlabel("# samples")
            plt.yscale("log",  nonposy='clip')
            ymin, ymax = plt.gca().get_ylim()
            plt.gca().set_ylim(ymin, ymax * 10)
            # plt.gca().set_ylim(0, 50)
            plt.savefig(path + "%d_dims_%s_dist_time.pdf" % (dims, dist), bbox_inches='tight', dpi=dpi)
            # plt.show()
            plt.close()

            if plot_mode == "bar":
                make_2d_barplot(data, "output_samples", "JSD", methods, bar_points=bar_points,
                                selector=["dims", "target_d"], selector_val=[dims, dist], labels=labels)
            else:
                make_2d_plot(data, "output_samples", "JSD", methods,
                             selector=["dims", "target_d"], selector_val=[dims, dist], labels=labels)
                plt.gca().set_xlim(bar_points[0], bar_points[-1])

            plt.gca().set_title("Target distribution: %s, Dimensions: %d" % (dist, dims))
            plt.gca().set_ylabel("Jensen-Shannon Divergence")
            plt.gca().set_xlabel("# samples")
            # plt.yscale("log")
            plt.savefig(path + "%d_dims_%s_dist_jsd.pdf" % (dims, dist), bbox_inches='tight', dpi=dpi)
            # plt.show()
            plt.close()

            if plot_mode == "bar":
                make_2d_barplot(data, "output_samples", "NESS", methods, bar_points=bar_points,
                                selector=["dims", "target_d"], selector_val=[dims, dist], labels=labels)
            else:
                make_2d_plot(data, "output_samples", "NESS", methods,
                             selector=["dims", "target_d"], selector_val=[dims, dist], labels=labels)
                plt.gca().set_xlim(bar_points[0], bar_points[-1])

            plt.gca().set_title("Target distribution: %s, Dimensions: %d" % (dist, dims))
            plt.gca().set_ylabel("Normalized Effective Sample Size")
            plt.gca().set_xlabel("# samples")
            ymin, ymax = plt.gca().get_ylim()
            plt.gca().set_ylim(0, min(1.2,ymax))
            # plt.yscale("log")
            plt.savefig(path + "%d_dims_%s_dist_ness.pdf" % (dims, dist), bbox_inches='tight', dpi=dpi)
            # plt.show()
            plt.close()

            if plot_mode == "bar":
                make_2d_barplot(data, "output_samples", "ev_mse", methods, bar_points=bar_points,
                                selector=["dims", "target_d"], selector_val=[dims, dist], labels=labels)
            else:
                make_2d_plot(data, "output_samples", "ev_mse", methods,
                             selector=["dims", "target_d"], selector_val=[dims, dist], labels=labels)
                plt.gca().set_xlim(bar_points[0], bar_points[-1])

            plt.gca().set_title("Target distribution: %s, Dimensions: %d" % (dist, dims))
            plt.gca().set_ylabel("Expected Value Mean Squared Error")
            plt.gca().set_xlabel("# samples")
            # plt.yscale("log")
            plt.savefig(path + "%d_dims_%s_dist_evmse.pdf" % (dims, dist), bbox_inches='tight', dpi=dpi)
            # plt.show()
            plt.close()

            print("Generated " + path + "%d_dims_%s_dist" % (dims, dist))


    # TODO: Get table at specific sample number values
    # TODO: Curate the data such that there is data points in the selected sample numbers

