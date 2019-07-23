import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches

def plot_sampled_pdfs(ax, samples, prob_p, shape=None, marginalize_axes = None):
    prob_p_margin = prob_p.reshape(shape)
    if marginalize_axes is not None:
        prob_p_margin = np.sum(prob_p_margin, axis=marginalize_axes)

    prob_p_margin_color = prob_p_margin / np.linalg.norm(prob_p_margin)
    ax.scatter(samples[:, 0], samples[:, 1], prob_p_margin, c=prob_p_margin_color, cmap=plt.cm.hot)

def plot_grid_sampled_pdfs(ax, dims, prob_p, shape=None, marginalize_axes = None):
    prob_p_margin = prob_p.reshape(shape).transpose()
    if marginalize_axes is not None:
        prob_p_margin = np.sum(prob_p_margin, axis=marginalize_axes)

    X = np.array(dims[0]).reshape(1,-1)
    Y = np.array(dims[1]).reshape(-1,1)

    # prob_p_margin_color = prob_p_margin / np.max(prob_p_margin)

    ax.plot_surface(X, Y, prob_p_margin, cmap=plt.cm.hot, linewidth=0, antialiased=True, shade=True, rstride=2, cstride=2, zorder=1)
    # ax.contour(X, Y, prob_p_margin.reshape(len(dims[0]),len(dims[1])), zdir='z', offset=-1.9, cmap=plt.cm.Reds)
    # ax.contour(X, Y, prob_p_margin.reshape(len(dims[0]),len(dims[1])), zdir='x', offset=-1, cmap=plt.cm.Reds)
    # ax.contour(X, Y, prob_p_margin.reshape(len(dims[0]),len(dims[1])), zdir='y', offset=1, cmap=plt.cm.Reds)

def plot_ellipsoids(axis, ellipsoids, points):
    color = iter("rgbkwcmy")
    for e in ellipsoids:
        c = next(color)
        axis.plot([e.loc[1]], [e.loc[0]], [-1.7], marker="v", alpha=1, c=c)
        axis.scatter(points[e.indices, 1], points[e.indices, 0], np.ones(np.count_nonzero(e.indices)) * -1.7, marker="o", alpha=1, c=c)

def plot_ellipsoids1D(axis, ellipsoids, points):
    color = iter("rgbkwcmy")
    for e in ellipsoids:
        c = next(color)
        axis.plot(e.loc[0], 0, marker="v", alpha=1, c=c)
        axis.plot(points[e.indices, 0], np.zeros(np.count_nonzero(e.indices)), marker="o", alpha=1, c=c)

def plot_tpyramid_area(axis, T):
    for n in T.leaves:
        x = n.center - n.radius
        height = n.value/(2*n.radius)
        rect = patches.Rectangle([x,0], n.radius*2, height, linewidth=0, linestyle="--", alpha=0.4, color=cm.cool(n.value*5))
        axis.add_patch(rect)

def plot_tpyramid_volume(axis, T):
    for n in T.leaves:
        min_coord = n.center - n.radius
        max_coord = n.center + n.radius
        z0 = 0
        z1 = n.value/((2*n.radius)**len(n.center))
        rect_prism(axis, [min_coord[0],max_coord[0]], [min_coord[1],max_coord[1]], [z0,z1])

def plot_grid_area(axis, samples, values, resolution):
    for s,v in zip(samples, values):
        x = s - resolution/2
        rect = patches.Rectangle([x,0], resolution, v/resolution, linewidth=0, linestyle="--", alpha=0.4, label="subvidided leaves")
        axis.add_patch(rect)

def plot_grid_volume(axis, samples, values, resolution):
    for s,v in zip(samples, values):
        min_coord = s - resolution / 2
        max_coord = s + resolution / 2
        z0 = 0
        z1 = v
        rect_prism(axis, [min_coord[0], max_coord[0]], [min_coord[1], max_coord[1]], [z0, z1])


# Draw rectangular prism. Obtained from: https://codereview.stackexchange.com/questions/155585/plotting-a-rectangular-prism
def rect_prism(ax, x_range, y_range, z_range):
    xx, yy = np.meshgrid(x_range, y_range)
    ax.plot_wireframe(xx, yy, z_range[0], color="r")
    ax.plot_surface(xx, yy, z_range[0], color="r", alpha=0.2)
    ax.plot_wireframe(xx, yy, z_range[1], color="r")
    ax.plot_surface(xx, yy, z_range[1], color="r", alpha=0.2)


    yy, zz = np.meshgrid(y_range, z_range)
    ax.plot_wireframe(x_range[0], yy, zz, color="r")
    ax.plot_surface(x_range[0], yy, zz, color="r", alpha=0.2)
    ax.plot_wireframe(x_range[1], yy, zz, color="r")
    ax.plot_surface(x_range[1], yy, zz, color="r", alpha=0.2)

    xx, zz = np.meshgrid(x_range, z_range)
    ax.plot_wireframe(xx, y_range[0], zz, color="r")
    ax.plot_surface(xx, y_range[0], zz, color="r", alpha=0.2)
    ax.plot_wireframe(xx, y_range[1], zz, color="r")
    ax.plot_surface(xx, y_range[1], zz, color="r", alpha=0.2)
