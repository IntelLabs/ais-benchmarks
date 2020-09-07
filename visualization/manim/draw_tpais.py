from manimlib.imports import *
from distributions.mixture.CGaussianMixtureModel import CGaussianMixtureModel
from sampling_methods.tree_pyramid import CTreePyramidSampling
import matplotlib.pyplot as plt


class CManimDistribution(GraphScene):
    CONFIG = {
        "x_min": -0.4,
        "x_max": 2,
        "x_axis_width": 12,
        "x_tick_frequency": 1,
        "x_leftmost_tick": None,  # Change if different from x_min
        "x_labeled_nums": np.arange(-0.4, 2, .2),
        "x_axis_label": "$x$",
        "x_decimal_number_config": {"num_decimal_places": 1},
        "y_min": 0.,
        "y_max": 2.,
        "y_axis_height": 5,
        "y_tick_frequency": 1,
        "y_bottom_tick": None,  # Change if different from y_min
        "y_labeled_nums": np.arange(0., 2., .2),
        "y_axis_label": "",
        "y_decimal_number_config": {"num_decimal_places": 1},
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

    def tpais_to_graph(self, x):
        return self.algo_tpais.model.prob(np.array([x]))

    def construct(self):
        self.dist = CGaussianMixtureModel({"means": np.array([[0.2], [1.2]]),
                                           "sigmas": np.array([[.01], [0.05]]),
                                           "support": np.array([-0.5, 1])})

        # Prepare parameters for inference algorithm
        inference_params = dict()
        inference_params["space_min"] = np.array([self.CONFIG["x_min"]])
        inference_params["space_max"] = np.array([self.CONFIG["x_max"]])
        inference_params["dims"] = 1
        inference_params["kde_bw"] = 0.01
        inference_params["method"] = "simple"
        inference_params["resampling"] = "none"
        inference_params["kernel"] = "haar"
        self.algo_tpais = CTreePyramidSampling(inference_params)

        x = np.linspace(inference_params["space_min"], inference_params["space_max"], 100)
        plt.plot(x.reshape(-1), self.dist.prob(x))

        # Show the target distribution
        self.setup_axes(animate=False)
        func_graph = self.get_graph(self.dist_to_graph, BLUE)
        self.play(ShowCreation(func_graph))

        # Show the proposal
        func_graph2 = self.get_graph(self.tpais_to_graph, RED).make_jagged()
        self.play(ShowCreation(func_graph2))

        posterior_samples = list()
        weights = list()

        for i in range(1, 20):
            n_inference_samples = 2
            posterior_samples, weights = self.algo_tpais.importance_sample(self.dist, n_inference_samples * i)

        # Lists of elements
        v_lines = list()
        samples_cross = list()
        samples_cross2 = list()

        # TODO: Find a way to show the resample process. Needs identifying which samples were resampled
        # TODO: Find a way to morph samples that were split into the new samples

        # Make geometries for the samples
        for s, w in zip(posterior_samples, weights):
            s_coord = self.coords_to_point(s, 0)
            # vert_line = self.get_vertical_line_to_graph(s, func_graph, color=YELLOW, stroke_width=2)
            vert_line = (Line(s_coord, s_coord + UP * w, color=YELLOW, stroke_width=2))
            v_lines.append(vert_line)
            samples_cross.append(Line(s_coord + LEFT*0.2 + DOWN*0.2, s_coord + RIGHT*0.2 + UP*0.2, stroke_width=2))
            samples_cross2.append(Line(s_coord + LEFT*0.2 + UP*0.2, s_coord + RIGHT*0.2 + DOWN*0.2, stroke_width=2))

        # Show animation of samples drawn from the proposal
        for l, x1, x2 in zip(v_lines, samples_cross, samples_cross2):
            self.play(ShowCreation(x1), ShowCreation(x2), run_time=.3)
            self.play(ScaleInPlace(x1, .5), ScaleInPlace(x2, .5), ShowCreation(l))

        # Show the adapted proposal
        self.play(Transform(func_graph2, self.get_graph(self.tpais_to_graph, RED).make_jagged()))

        self.algo_tpais.draw(plt.gca())
        # Draw samples from the TPAIS proposal after posterior inference.
        plt.pause(0.001)
        plt.gca().scatter(list(posterior_samples), [-0.4] * len(posterior_samples), marker="x", c='g', label="TP-AIS: Samples")

        plt.gca().set_xlim(inference_params["space_min"], inference_params["space_max"])
        plt.legend(scatterpoints=1)

        plt.pause(0.001)
        plt.show(block=True)


if __name__ == "__main__":
    import os
    os.system("manim draw_tpais.py CManimDistribution -pl")
