import numpy as np

from .base import BaseDynamics


class SingleIntegrator2D(BaseDynamics):
    """Planar single-integrator dynamics with state [x, y]."""

    state_dim = 2
    control_dim = 2

    def derivative(self, state, control):
        return np.asarray(control, dtype=float).reshape(self.control_dim, 1)
