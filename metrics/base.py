"""
TODO: Write proper unit testing for all the metrics implemented.
"""


class CMetric:
    def __init__(self):
        self.name = "base"
        self.type = "base"

    def compute(self, params):
        raise NotImplementedError

