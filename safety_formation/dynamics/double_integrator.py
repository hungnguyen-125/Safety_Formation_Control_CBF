import numpy as np

from .base import BaseDynamics


class DoubleIntegrator2D(BaseDynamics):
    """Planar double-integrator dynamics with state [x, y, vx, vy]."""

    state_dim = 4
    control_dim = 2

    A = np.array(
        [
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=float,
    )

    B = np.array(
        [
            [0, 0],
            [0, 0],
            [1, 0],
            [0, 1],
        ],
        dtype=float,
    )

    def derivative(self, state, control):
        state = np.asarray(state, dtype=float).reshape(self.state_dim, 1)
        control = np.asarray(control, dtype=float).reshape(self.control_dim, 1)
        return self.A @ state + self.B @ control

    def step(self, state, control, dt):
        x, y, vx, vy = np.asarray(state, dtype=float).reshape(4)
        ax, ay = np.asarray(control, dtype=float).reshape(2)

        return np.array([
            [x  + vx * dt + 0.5 * ax * dt**2],
            [y  + vy * dt + 0.5 * ay * dt**2],
            [vx + ax * dt],
            [vy + ay * dt],
        ])