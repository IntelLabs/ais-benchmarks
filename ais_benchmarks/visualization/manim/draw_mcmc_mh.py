from manimlib.imports import *
from distributions.mixture.CGaussianMixtureModel import CGaussianMixtureModel
from distributions.parametric.CMultivariateNormal import CMultivariateNormal
from sampling_methods.metropolis_hastings import CMetropolisHastings


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

    def mcmc_to_graph(self, x):
        if len(self.algo_mcmc.samples):
            dist = CMultivariateNormal({"mean": self.algo_mcmc.samples[-1], "sigma": self.algo_mcmc.proposal_d.cov})
        else:
            dist = self.algo_mcmc.proposal_d
        return dist.prob(np.array([x]))

    def construct(self):
        self.dist = CGaussianMixtureModel({"means": np.array([[0.2], [1.2]]),
                                           "sigmas": np.array([[.01], [0.05]]),
                                           "support": np.array([-0.5, 1])})

        # Prepare parameters for inference algorithm
        inference_params = dict()
        inference_params["space_min"] = np.array([self.CONFIG["x_min"]])
        inference_params["space_max"] = np.array([self.CONFIG["x_max"]])
        inference_params["dims"] = 1
        inference_params["n_steps"] = 1
        inference_params["n_burnin"] = 0
        inference_params["proposal_sigma"] = .1
        inference_params["proposal_d"] = None
        inference_params["kde_bw"] = 0.01
        self.algo_mcmc = CMetropolisHastings(inference_params)

        # Show the target distribution
        self.setup_axes(animate=False)
        func_graph = self.get_graph(self.dist_to_graph, BLUE)
        graph_lab = self.get_graph_label(func_graph, label="\\pi(x)")
        graph_lab.set_y((2.5 * DOWN + 4 * LEFT)[1] - .5)
        self.play(ShowCreation(func_graph))
        self.play(ShowCreation(graph_lab))

        # Show the proposal
        func_graph2 = self.get_graph(self.mcmc_to_graph, GREEN)
        graph2_lab = self.get_graph_label(func_graph2, label="Q(x)")
        graph2_lab.set_y((2.5 * DOWN + 4 * LEFT)[1] - 1.1)
        self.play(ShowCreation(func_graph2))
        self.play(ShowCreation(graph2_lab))

        last_traj_sample = 0
        s_last = 0
        for i in range(0, 10):
            n_inference_samples = 1
            posterior_samples, weights = self.algo_mcmc.importance_sample(self.dist, n_inference_samples * i)

            # Make geometries for the samples
            for s, s_type in zip(self.algo_mcmc.trajectory_samples[last_traj_sample-1:], self.algo_mcmc.trajectory_types[last_traj_sample-1:]):
                print("sample: ", s, "type: ", s_type, "last: ", s_last)
                s_coord = self.coords_to_point(s, 0)
                s_last_coord = self.coords_to_point(s_last, 0)

                # Arrow from the last accepted sample. The extra code is to force the arrow always curve upwards
                if s_last_coord[0] > s_coord[0]:
                    arrow = CurvedArrow(s_last_coord + UP * .1, s_coord + UP * .1, color=WHITE, stroke_width=2, tip_length=.15)
                else:
                    arrow = CurvedArrow(s_coord + UP * .1, s_last_coord + UP * .1, color=WHITE, stroke_width=2, tip_length=.15)
                    arrow.flip()

                # Display proposal
                color = WHITE
                x_hat1 = Line(s_coord + LEFT * 0.1 + DOWN * 0.1, s_coord + RIGHT * 0.1 + UP * 0.1, stroke_width=3,
                              color=color)
                x_hat2 = Line(s_coord + LEFT * 0.1 + UP * 0.1, s_coord + RIGHT * 0.1 + DOWN * 0.1, stroke_width=3,
                              color=color)
                self.play(ShowCreation(arrow), ShowCreation(x_hat1), ShowCreation(x_hat2), run_time=.3)

                # Show both probabilities of x_old and x_hat
                line_px_hat = self.get_vertical_line_to_graph(s, func_graph, color=RED)
                line_px_old = self.get_vertical_line_to_graph(s_last, func_graph, color=GREEN)

                if last_traj_sample == 1:
                    # Show both probabilities sequentially
                    self.play(ShowCreation(line_px_hat))
                    self.play(ShowCreation(line_px_old))

                    # Show acceptance formula
                    formula = TexMobject(r"A(x,x') =", r" min\left(1, ", r"\frac{\pi(x')Q(x|x')}{\pi(x)Q(x'|x)}\right)")
                    formula.shift(UP * 3)
                    formula.shift(RIGHT * 2)
                    self.play(Write(formula), run_time=3.)

                    # Simplify it because we are using a symmetric proposal and Q(x'|x) = Q(x|x')
                    formula2 = TexMobject(r"A(x,x') =", r" min\left(1, ", r"\frac{\pi(x')}{\pi(x)}\right)")
                    formula2.shift(UP * 3)
                    formula2.shift(RIGHT * .5)
                    self.play(Transform(formula, formula2), run_time=3.)
                else:
                    self.play(ShowCreation(line_px_hat), ShowCreation(line_px_old))

                if last_traj_sample > 0:
                    # Display current sample and proposal probabilities
                    px_old = self.point_to_coords(line_px_old.end)[1]
                    px_hat = self.point_to_coords(line_px_hat.end)[1]
                    px_old_txt = TexMobject("\\pi(x) = %4.2f" % px_old, color=GREEN)
                    px_hat_txt = TexMobject("\\pi(x')= %4.2f" % px_hat, color=RED)
                    px_old_txt.shift(UP * 1.4)
                    px_old_txt.shift(RIGHT * 2)
                    px_hat_txt.shift(UP * 2)
                    px_hat_txt.shift(RIGHT * 2)
                    self.play(Transform(line_px_hat, px_hat_txt), Transform(line_px_old, px_old_txt))

                    # Display acceptance prob
                    A = px_hat / px_old
                    A_result = TexMobject(" = %5.3f" % min(1.0, A), color=WHITE)
                    A_result.shift(UP * 3)
                    A_result.shift(RIGHT * 4.5)
                    self.play(Write(A_result))

                    # TODO: Animate generation of a random value and the accept/reject decision
                    # a = np.random.uniform()

                    # Display accept or reject
                    if s_type == self.algo_mcmc.ACCEPT or s_type == self.algo_mcmc.SAMPLE:
                        result_txt = TexMobject(r"\text{Accept x'}", color=GREEN)
                    else:
                        result_txt = TexMobject(r"\text{Reject x'}", color=RED)

                    result_txt.shift(UP * 2)
                    result_txt.shift(RIGHT * 5)
                    self.play(Write(result_txt))

                # Display proposal as accepted or rejected
                if s_type == self.algo_mcmc.ACCEPT or s_type == self.algo_mcmc.SAMPLE:
                    color = GREEN
                    s_last = s
                elif s_type == self.algo_mcmc.BURN_IN:
                    color = BLUE
                elif s_type == self.algo_mcmc.REJECT:
                    color = RED
                elif s_type == self.algo_mcmc.DECORRELATION:
                    color = RED

                self.play(Transform(x_hat1, Line(s_coord + LEFT * 0.1 + DOWN * 0.1, s_coord + RIGHT * 0.1 + UP * 0.1, stroke_width=2, color=color)),
                          Transform(x_hat2, Line(s_coord + LEFT * 0.1 + UP * 0.1, s_coord + RIGHT * 0.1 + DOWN * 0.1, stroke_width=2, color=color)))

                self.play(ScaleInPlace(x_hat1, .5), ScaleInPlace(x_hat2, .5))

                # Clean up arrow
                self.remove(arrow)
                self.remove(line_px_hat)
                self.remove(line_px_old)
                if last_traj_sample > 0:
                    self.remove(result_txt)
                    self.remove(A_result)
                # if last_traj_sample == 1:
                #     self.remove(formula)

                last_traj_sample += 1

            # Show the adapted proposal
            self.play(Transform(func_graph2, self.get_graph(self.mcmc_to_graph, GREEN)), run_time=.5)


if __name__ == "__main__":
    import os
    # os.system("manim draw_mcmc_mh.py CManimDistribution -pl")
    os.system("manim draw_mcmc_mh.py CManimDistribution -p")
