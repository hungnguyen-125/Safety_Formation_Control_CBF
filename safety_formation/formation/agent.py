import numpy as np

class Agent:
    """
    Agent class representing an individual entity in a Multi-Agent System.
    The dynamics follow a Double Integrator model: x_dot = A x + B u.
    """

    # System matrices A and B (class attributes shared by all agents)
    # State vector: x = [x, y, vx, vy]^T (4x1)
    # Control input: u = [ax, ay]^T (2x1)
    A = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ], dtype=float)

    B = np.array([
        [0, 0],
        [0, 0],
        [1, 0],
        [0, 1]
    ], dtype=float)

    def __init__(self, agent_id, x0, f_target):
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

        # Ensure state and formation offset are column vectors (4,1)
        self.state = np.array(x0, dtype=float).reshape(4, 1)
        self.f = np.array(f_target, dtype=float).reshape(4, 1)

        # Store state history (similar to Simulink "To Workspace" block)
        self.history = []

    def update_physics(self, u, dt):
        """
        Update the agent physical state using Euler integration.

        x(t+1) = x(t) + x_dot * dt
        x(t+1) = x(t) + (A x(t) + B u(t)) dt
        """
        # Ensure control input is a column vector (2,1)
        u_input = np.array(u, dtype=float).reshape(2, 1)

        # Compute state derivative
        x_dot = self.A @ self.state + self.B @ u_input

        # Update state
        self.state = self.state + x_dot * dt

        # Save updated state to history
        self.save_history()

    def save_history(self):
        """Store a copy of the current state (flattened) for plotting."""
        self.history.append(self.state.flatten())

    @property
    def pos(self):
        """Return current position vector [x, y]^T (2x1)."""
        return self.state[0:2]

    @property
    def vel(self):
        """Return current velocity vector [vx, vy]^T (2x1)."""
        return self.state[2:4]

    def get_full_history(self):
        """Convert history list to numpy array for convenient slicing."""
        return np.array(self.history)

    def __repr__(self):
        return f"Agent(id={self.id}, pos={self.pos.flatten()})"