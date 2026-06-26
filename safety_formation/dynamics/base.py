import numpy as np


class BaseDynamics:
    """Base interface for agent dynamics models."""

    state_dim = None
    control_dim = None

    def derivative(self, state, control):
        raise NotImplementedError("Subclasses must implement derivative")

    def step(self, state, control, dt):
        state = np.asarray(state, dtype=float).reshape(self.state_dim, 1)
        control = np.asarray(control, dtype=float).reshape(self.control_dim, 1)
        return state + self.derivative(state, control) * dt

    def position(self, state):
        state = np.asarray(state, dtype=float).reshape(self.state_dim, 1)
        return state[0:2]

    def velocity(self, state):
        state = np.asarray(state, dtype=float).reshape(self.state_dim, 1)
        if self.state_dim < 4:
            return np.zeros((2, 1))
        return state[2:4]
