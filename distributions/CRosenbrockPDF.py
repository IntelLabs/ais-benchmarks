import numpy as np

class CRosenbrock:
    def __init__(self,a=1, b=100):
        self.a = a
        self.b = b

    def log_prob(self, x):
        if len(x.shape) == 1:
            x = x.reshape(1,-1)

        res = np.zeros(len(x))

        for i in range(len(x[0])-1):
            xi = x[:,i]
            xi1 = x[:,i + 1]
            res = res + self.b * (xi1-xi*xi) * (xi1-xi*xi) + (self.a - xi)*(self.a - xi)

        return np.log(1/res)