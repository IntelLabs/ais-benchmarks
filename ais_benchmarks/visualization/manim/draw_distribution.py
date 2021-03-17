from manimlib.imports import *
from distributions.mixture.CGaussianMixtureModel import CGaussianMixtureModel
from distributions.parametric.CMultivariateNormal import CMultivariateNormal


class CManimDistribution(GraphScene):
    CONFIG = {
        "x_min": -1,
        "x_max": 8,
        "x_axis_width": 9,
        "x_tick_frequency": 1,
        "x_leftmost_tick": None,  # Change if different from x_min
        "x_labeled_nums": range(-1, 8, 1),
        "x_axis_label": "$x$",
        "y_min": 0,
        "y_max": 1,
        "y_axis_height": 6,
        "y_tick_frequency": .1,
        "y_bottom_tick": None,  # Change if different from y_min
        "y_labeled_nums": np.arange(0., 1., .1),
        "y_axis_label": "$p(x)$",
        "axes_color": GREY,
        "graph_origin": 2.5 * DOWN + 4 * LEFT,
        "exclude_zero_label": True,
        "default_graph_colors": [BLUE, GREEN, YELLOW],
        "default_derivative_color": GREEN,
        "default_input_color": YELLOW,
        "default_riemann_start_color": BLUE,
        "default_riemann_end_color": GREEN,
        "area_opacity": 0.8,
        "num_rects": 50,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def dist_to_graph(self, x):
        return self.dist.prob(np.array([x]))

    def construct(self):
        self.dist = CGaussianMixtureModel({"means": np.array([[.2], [4]]),
                                           "sigmas": np.array([[.1], [1]]),
                                           "support": np.array([-5, 10])})

        self.setup_axes(animate=False)
        func_graph = self.get_graph(self.dist_to_graph, "BLUE")
        # vert_line = self.get_vertical_line_to_graph(TAU, func_graph, color=YELLOW)
        # graph_lab = self.get_graph_label(func_graph, label="\\pi(x)")
        # label_coord = self.input_to_graph_point(TAU, func_graph)

        self.play(ShowCreation(func_graph))
        # self.play(ShowCreation(vert_line), ShowCreation(graph_lab), ShowCreation(graph_lab2), ShowCreation(two_pi))


if __name__ == "__main__":
    import os
    os.system("manim draw_distribution.py CManimDistribution -p")