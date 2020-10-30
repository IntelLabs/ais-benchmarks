
class CMetric:
    def __init__(self):
        self.name = "base"
        self.type = "base"
        self.value = 0

    def compute(self, **kwargs):
        raise NotImplementedError

    def pre(self, **kwargs):
        raise NotImplementedError

    def post(self, **kwargs):
        raise NotImplementedError

    def reset(self, **kwargs):
        raise NotImplementedError
