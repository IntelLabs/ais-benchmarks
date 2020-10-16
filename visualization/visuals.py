import numpy as np


class CColor:
    WHITE = np.array([1, 1, 1, 1])
    BLACK = np.array([0, 0, 0, 1])
    RED = np.array([1, 0, 0, .7])
    GREEN = np.array([0, .7, 0, 1])
    BLUE = np.array([0, 0, .7, 1])


class CVisual:
    def __init__(self, id):
        self.id = id
        self.name = None
        self.type = None  # ["axis", "function", "sample", "expression"]
        self.outline_color = CColor.BLACK
        self.fill_color = CColor.BLACK
        self.pos = np.array([0, 0, 0])
        self.draw_style = "-"  # ["-", "--", ".", "x"] solid, dashed, point, cross


#######################################################################################################################
# FUNCTION VISUALIZATION ELEMENTS
#######################################################################################################################
class CAxis(CVisual):
    def __init__(self, id, start=np.array([0, 0, 0]), end=np.array([1, 0, 0])):
        super().__init__(id)
        self.type = "axis"      # Visual component type is fixed to axis type
        self.ticks = []         # List with the axis values that have a visible tick
        self.ticks_size = []    # Size of each tick, must have the same size as self.ticks
        self.ticks_lbl = []     # Label shown on each tick. Must have the same size as self.ticks, use None to not show.
        self.start = start      # Axis initial position (x, y, z)
        self.end = end          # Axis final position (x, y, z)

        # Default tick configuration
        self.ticks = np.linspace(0, np.linalg.norm(self.end - self.start), num=10, endpoint=True)
        self.ticks_lbl = ["%3.1f" % t_val for t_val in self.ticks]
        self.ticks_size = [np.linalg.norm(self.end - self.start) * .01] * len(self.ticks)


class CFunction(CVisual):
    def __init__(self, id, func, limits, resolution=1000):
        super().__init__(id)
        self.type = "function"
        self.func = func
        self.limits = limits
        self.resolution = resolution
        self.xs = np.linspace(limits[0], limits[1], num=resolution)
        self.ys = self.func(self.xs)


class CProbabilityDensityFunction(CFunction):
    def __init__(self, id, func, limits, resolution=1000):
        super().__init__(id, func, limits, resolution)


class CTargetDist(CProbabilityDensityFunction):
    def __init__(self, id, func, limits, resolution=1000):
        super().__init__(id, func, limits, resolution)
        self.draw_style = "-"
        self.outline_color = CColor.RED
        self.fill_color = CColor.RED


class CProposalDist(CProbabilityDensityFunction):
    def __init__(self, id, func, limits, resolution=1000):
        super().__init__(id, func, limits, resolution)
        self.draw_style = "-"
        self.outline_color = CColor.GREEN
        self.fill_color = CColor.GREEN


class CProposalDistComponent(CProbabilityDensityFunction):
    def __init__(self, id, func, weight, limits, resolution=1000):
        self.func_orig = func
        self.weight = weight
        super().__init__(id, self.func_w, limits, resolution)
        self.draw_style = "--"
        self.outline_color = CColor.GREEN
        self.fill_color = CColor.GREEN

    def func_w(self, x):
        return self.func_orig(x) * self.weight


#######################################################################################################################
# SAMPLE VISUALIZATION
#######################################################################################################################
class CPoint(CVisual):
    def __init__(self, id, pos, draw_style="."):
        super().__init__(id)
        self.type = "point"
        self.pos = pos
        self.draw_style = draw_style


class CProposedSample(CPoint):
    def __init__(self, id, pos):
        super().__init__(id, pos, draw_style="x")
        self.outline_color = CColor.BLUE
        self.fill_color = CColor.BLUE


class CAcceptedSample(CPoint):
    def __init__(self, id, pos):
        super().__init__(id, pos, draw_style="o")
        self.outline_color = CColor.GREEN
        self.fill_color = CColor.GREEN


class CRejectedSample(CPoint):
    def __init__(self, id, pos):
        super().__init__(id, pos, draw_style="x")
        self.outline_color = CColor.RED
        self.fill_color = CColor.RED


class CImportanceSample(CPoint):
    def __init__(self, id, pos, weight):
        super().__init__(id, pos, draw_style="o")
        self.outline_color = CColor.GREEN
        self.fill_color = CColor.GREEN
        self.w = weight


#######################################################################################################################
# EXPRESSION AND LABEL VISUALIZATION
#######################################################################################################################
class CLabel(CVisual):
    def __init__(self, id, pos, text):
        super().__init__(id)
        self.pos = pos
        self.text = text


class CExpression(CLabel):
    def __init__(self, id, pos, text):
        super().__init__(id, pos, text)


#######################################################################################################################
# OTHER RELEVANT COMPONENTS
#######################################################################################################################
class CBox(CVisual):
    def __init__(self, id, limits):
        super().__init__(id)
        self.limits = limits


#######################################################################################################################
# EXAMPLES AND TESTS
#######################################################################################################################
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x_axis = CAxis(id=0, start=np.array([-1, 0, 0]), end=np.array([1, 0, 0]))
    y_axis = CAxis(id=1, end=np.array([0, 1, 0]))

    function = CFunction(id=2, limits=[0, 1], func=lambda x: .5+.4*np.sin(10*x))
    function.outline_color = CColor.BLUE

    prop_sample = CProposedSample(id=3, pos=[.2, .2, 0])
    acc_sample = CAcceptedSample(id=4, pos=[.3, .2, 0])
    rej_sample = CRejectedSample(id=5, pos=[.2, .3, 0])
    imp_sample = CImportanceSample(id=6, pos=[.4, .2, 0], weight=.3)

    viz_elements = [x_axis, y_axis, function, prop_sample, acc_sample, rej_sample, imp_sample]

    from visualization.matplotlib.viz_interface import draw_sequence

    draw_sequence(viz_elements)

    plt.xlim(-.1, 1.1)
    plt.ylim(-.1, 1.1)
    plt.show()
