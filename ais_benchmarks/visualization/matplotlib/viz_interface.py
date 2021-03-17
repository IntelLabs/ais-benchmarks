import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np


def draw_point(item, ax):
    return ax.scatter(item.pos[0], item.pos[1], marker=item.draw_style, color=item.outline_color)


def draw_func(item, ax):
    return ax.plot(item.xs, item.ys, color=item.outline_color, linestyle=item.draw_style)


def draw_axis(item, ax):
    xmin = item.pos[0] + item.start[0]
    xmax = item.pos[0] + item.end[0]
    ymin = item.pos[1] + item.start[1]
    ymax = item.pos[1] + item.end[1]
    elems = list()
    l = mlines.Line2D([xmin, xmax], [ymin, ymax], color=item.outline_color)
    ax.add_line(l)
    elems.append(l)

    for (t, t_size, t_lbl) in zip(item.ticks, item.ticks_size, item.ticks_lbl):
        # Get the normal direction to the axis
        dir = (item.end - item.start) / np.linalg.norm(item.end - item.start)
        dir_normal = np.array([-dir[1], dir[0], dir[2]])

        # Obtain tick initial and final points
        ini = t * dir - dir_normal * t_size + item.start
        end = t * dir + dir_normal * t_size + item.start

        # Draw a small line perpendicular to the axis with size t_size
        l = mlines.Line2D([ini[0], end[0]], [ini[1], end[1]], color=item.outline_color)
        ax.add_line(l)

        # Display label
        # Find the direction that puts the labels outwards
        if (t * dir)[0] > (t * dir)[1]:
            text_pos = t * dir - dir_normal * 4 * t_size + item.start
        else:
            text_pos = t * dir + dir_normal * 4 * t_size + item.start

        txt = plt.text(text_pos[0], text_pos[1], t_lbl, horizontalalignment="center", verticalalignment="center", color=item.outline_color)
        elems.append(txt)

    elems.append(l)

    return elems


def draw_item(item, ax):
    if item.type == "axis":
        return draw_axis(item, ax)
    elif item.type == "function":
        return draw_func(item, ax)
    elif item.type == "point":
        return draw_point(item, ax)
    else:
        raise NotImplementedError("%s visual item type not implemented" % item.type)


def draw_frames(frames, static_elems):
    for f in frames:
        plt.clf()
        plt.cla()
        plt.gca().axis("off")
        draw_sequence(static_elems)
        draw_sequence(f)
        plt.pause(0.1)


def draw_sequence(seq):
    items = list()
    object_idxs = list()
    for it in seq:
        # Remove/morph elements with indices that are already drawn
        try:
            idx = object_idxs.index(it.id)
            for elem in items[idx]:
                elem.remove()
            items[idx] = draw_item(item=it, ax=plt.gca())
        except ValueError:
            object_idxs.append(it.id)
            items.append(draw_item(item=it, ax=plt.gca()))
