import numpy as np


class ZeroDisturbance:
    def apply(self, control, step, time, agents):
        return np.asarray(control, dtype=float)


class ConstantControlDisturbance:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=float)

    def apply(self, control, step, time, agents):
        return np.asarray(control, dtype=float) + self.value
