import numpy as np
from numpy import array as t_tensor

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches


def make_grid(space_min, space_max, resolution):
    dim_range = space_max - space_min
    num_samples = (dim_range / resolution).tolist()
    for i in range(len(num_samples)):
        num_samples[i] = int(num_samples[i])
        if num_samples[i] < 1:
            num_samples[i] = 1

    dimensions = []
    shape = t_tensor([0] * len(num_samples))
    for i in range(len(num_samples)):
        dimensions.append(np.linspace(space_min[i], space_max[i], num_samples[i]))
        shape[i] = len(dimensions[i])

    samples = np.array(np.meshgrid(*dimensions)).T.reshape(-1, len(space_min))
    return samples, dimensions, shape


def grid_sample_distribution(dist, space_min, space_max, resolution):
    grid, dims, shape = make_grid(space_min, space_max, resolution)
    prob = dist.prob(t_tensor(grid))
    return grid, prob, dims, shape


def plot_pdf(ax, pdf, space_min, space_max, resolution=0.1, options="-b", alpha=0.2, scale=1.0, label=None):
    x = np.linspace(space_min, space_max, int((space_max - space_min) / resolution)).reshape(-1, 1)
    y = pdf.prob(x) * scale
    return ax.plot(x.flatten(), y.flatten(), options, alpha=alpha, label=label)


def plot_sampled_pdfs(ax, samples, prob_p, shape=None, marginalize_axes = None):
    prob_p_margin = prob_p.reshape(shape)
    if marginalize_axes is not None:
        prob_p_margin = np.sum(prob_p_margin, axis=marginalize_axes)

    prob_p_margin_color = prob_p_margin / np.linalg.norm(prob_p_margin)
    ax.scatter(samples[:, 0], samples[:, 1], prob_p_margin, c=prob_p_margin_color, cmap=plt.cm.hot)


def plot_grid_sampled_pdfs(ax, dims, prob_p, shape=None, marginalize_axes=None, alpha=1.0, cmap=plt.cm.hot, label=None, linestyles='solid'):
    prob_p_margin = prob_p.reshape(shape).transpose()
    if marginalize_axes is not None:
        prob_p_margin = np.sum(prob_p_margin, axis=marginalize_axes)

    # Format X and Y for the surface plot
    # X = np.array(dims[0]).reshape(1, -1)
    # Y = np.array(dims[1]).reshape(-1, 1)

    # prob_p_margin_color = prob_p_margin / np.max(prob_p_margin)

    # return ax.plot_surface(X, Y, prob_p_margin, cmap=cmap, linewidth=0, antialiased=True, shade=True, rstride=2,
    #                        cstride=2, zorder=1, alpha=alpha, label=label)
    # return ax.plot_surface(X, Y, prob_p_margin, cmap=cmap, linewidth=0, antialiased=True, shade=True, rstride=2,
    #                        cstride=2, alpha=alpha, label=label)
    levels = np.arange(np.min(prob_p_margin), np.max(prob_p_margin), (np.max(prob_p_margin)-np.min(prob_p_margin)) / 15)
    CS = ax.contour(dims[0], dims[1], prob_p_margin.reshape(len(dims[0]), len(dims[1])), zdir='z', offset=0, cmap=cmap, levels=levels, linestyles=linestyles, alpha=alpha)
    return CS.collections if len(CS.collections) else []
    # ax.contour(X, Y, prob_p_margin.reshape(len(dims[0]),len(dims[1])), zdir='x', offset=-1, cmap=plt.cm.Reds)
    # ax.contour(X, Y, prob_p_margin.reshape(len(dims[0]),len(dims[1])), zdir='y', offset=1, cmap=plt.cm.Reds)


def plot_pdf2d(ax, pdf, space_min, space_max, resolution=0.1, alpha=0.2, scale=1.0, label=None, colormap=plt.cm.cool, linestyles='solid'):
    grid, prob, dims, shape = grid_sample_distribution(pdf, space_min, space_max, resolution=resolution)
    return plot_grid_sampled_pdfs(ax, dims, prob * scale, shape=shape, alpha=alpha, cmap=colormap, label=label, linestyles=linestyles)


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


def plot_tpyramid_area(axis, T, scale=1, label=None):
    rects = []
    for i, n in enumerate(T.leaves):
        x = n.center - n.radius

        heightw = n.weight * scale

        # area
        height = n.weight * n.sampler.prob(n.center) * scale
        color = cm.cool(height).flatten()
        rect = patches.Rectangle([x, 0], n.radius*2, height.flatten(), linewidth=1, linestyle="--", alpha=0.4, color=color)
        rects.append(axis.add_patch(rect))

        # weights
        rect = patches.Rectangle([x, 0], n.radius * 2, heightw, linewidth=0, linestyle="--", alpha=0.8, color="r")
        rects.append(axis.add_patch(rect))

    rect.set_label(label)
    return rects


def plot_tpyramid_weights(axis, T, scale=1, label=None):
    rects = []
    for i, n in enumerate(T.leaves):
        x = n.center - n.radius
        # height = n.weight * n.sampler.prob(n.center) * scale
        height = n.weight * scale
        rect = patches.Rectangle([x,0], n.radius*2, height.flatten(), linewidth=0, linestyle="--", alpha=0.4, color="r")
        rects.append(axis.add_patch(rect))
    rect.set_label(label)
    return rects


def plot_tpyramid_volume(axis, T):
    for n in T.leaves:
        min_coord = n.center - n.radius
        max_coord = n.center + n.radius
        z0 = 0
        height = n.weight * n.sampler.prob(n.center)
        rect_prism(axis, [min_coord[0],max_coord[0]], [min_coord[1],max_coord[1]], [z0,height])


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
