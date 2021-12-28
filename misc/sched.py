import numpy as np


class Scheduler(object):
    def __init__(self, max_time, init, final):
        self.init = init
        self.final = final
        self.current = init
        self.max_time = max_time

    def __call__(self, *args, **kwargs):
        pass


class SigmoidScheduler(Scheduler):
    def __init__(self, max_time, init=-10, final=10, scale=1):
        super().__init__(max_time, init, final)
        self.ratio = (self.final - self.init)/self.max_time
        self.scale = scale

    def __call__(self, *args, **kwargs):
        self.current += self.ratio
        return self.scale * self.sigmoid(self.current)

    @staticmethod
    def sigmoid(x):
        return 1/(1 + np.exp(-x))

