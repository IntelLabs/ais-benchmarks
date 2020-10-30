
class CMetric:
    def __init__(self):
        self.name = "base"
        self.type = "base"

    def compute(self, params):
        raise NotImplementedError

