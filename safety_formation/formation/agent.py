import numpy as np

from safety_formation.dynamics import DoubleIntegrator2D


class Agent:
    """
    Agent class representing an individual entity in a Multi-Agent System.

    By default, the dynamics follow a double-integrator model:
    x_dot = A x + B u.
    """

    A = DoubleIntegrator2D.A
    B = DoubleIntegrator2D.B

    def __init__(self, agent_id, x0, f_target, alpha=10, beta=10, dynamics=None):
        """
        Initialize an Agent.

        Parameters
        ----------
        agent_id : int
            Unique identifier of the agent.
        x0 : array-like
            Initial state [x, y, vx, vy].
        f_target : array-like
            Desired relative formation offset [fx, fy, 0, 0].
        """
        self.id = agent_id
        self.dynamics = dynamics if dynamics is not None else DoubleIntegrator2D()

        # Ensure state and formation offset are column vectors.
        self.state = np.array(x0, dtype=float).reshape(self.dynamics.state_dim, 1)
        self.f = np.array(f_target, dtype=float).reshape(self.dynamics.state_dim, 1)

        self.alpha = alpha
        self.beta = beta
        
        # Store state history (similar to Simulink "To Workspace" block)
        self.history = []

    def update_physics(self, u, dt):
        """
        Update the agent physical state using Euler integration.

        x(t+1) = x(t) + x_dot * dt
        x(t+1) = x(t) + (A x(t) + B u(t)) dt
        """
        self.state = self.dynamics.step(self.state, u, dt)

        # Save updated state to history
        self.save_history()

    def save_history(self):
        """Store a copy of the current state (flattened) for plotting."""
        self.history.append(self.state.flatten())

    def clear_history(self):
        self.history = []
        
    @property
    def pos(self):
        """Return current position vector [x, y]^T (2x1)."""
        return self.dynamics.position(self.state)

    @property
    def vel(self):
        """Return current velocity vector [vx, vy]^T (2x1)."""
        return self.dynamics.velocity(self.state)

    def get_full_history(self):
        """Convert history list to numpy array for convenient slicing."""
        return np.array(self.history)

    def __repr__(self):
        return f"Agent(id={self.id}, pos={self.pos.flatten()})"
