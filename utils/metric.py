import numpy as np

class ErrorTracker:
    """
    Track element-wise absolute tracking error:
        e_i = |x_i - f_i - x0|
    where x0 is leader state (4x1). Followers are 1..N.
    """

    def __init__(self):
        self.e_abs_hist = []     # list of (4N,) vectors over time
        self.e_abs_mean = []     # scalar mean(|e|)
        self.e_abs_max = []      # scalar max(|e|)

    def step(self, agents, leader_state):
        """
        Compute and store absolute error for current time step.

        Parameters
        ----------
        agents : list[Agent]
            Followers list.
        leader_state : array-like (4,1) or (4,)
            Leader state x0.
        """
        x0 = np.array(leader_state, dtype=float).reshape(4, 1)

        # Stack |x_i - f_i - x0| for all followers -> (4N,1)
        E_abs = np.vstack([np.abs(ag.state - ag.f - x0) for ag in agents])

        e_flat = E_abs.flatten()  # (4N,)
        self.e_abs_hist.append(e_flat)
        self.e_abs_mean.append(float(np.mean(e_flat)))
        self.e_abs_max.append(float(np.max(e_flat)))

    def get_history(self):
        """Return abs-error history as array (T, 4N)."""
        return np.array(self.e_abs_hist)

    def get_mean_history(self):
        """Return mean(|e|) per step as array (T,)."""
        return np.array(self.e_abs_mean)

    def get_max_history(self):
        """Return max(|e|) per step as array (T,)."""
        return np.array(self.e_abs_max)