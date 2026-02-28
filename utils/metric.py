import numpy as np

def _as_state4(x):
    """Convert input to shape (4,1). Accepts Agent (has .state) or array-like."""
    if hasattr(x, "state"):
        x = x.state
    return np.array(x, dtype=float).reshape(4, 1)


def compute_delta(agents, leader, use_pos_only=False):
    """
    Compute stacked formation/tracking error over time:

        delta_i(t) = x_i(t) - f_i - x0(t)

    Parameters
    ----------
    agents : list[Agent]
        Followers. Each agent must have:
            - history: list of flattened state arrays (T,4)
            - f: formation offset (4,1)
    leader : Agent or array-like
        Leader agent (with .history) OR a constant leader state (4,) or (4,1).
        - If Agent: uses leader.history over time.
        - If array-like: assumes leader is constant over time.
    use_pos_only : bool
        If True, use only position components [x,y].

    Returns
    -------
    delta : np.ndarray
        If use_pos_only=False: shape (T, N, 4)
        If use_pos_only=True : shape (T, N, 2)
    """
    if len(agents) == 0:
        raise ValueError("agents list is empty")

    T = len(agents[0].history)
    for ag in agents:
        if len(ag.history) != T:
            raise ValueError("All agents must have the same history length")

    # leader history or constant leader state
    if hasattr(leader, "history"):
        leader_hist = np.array(leader.history, dtype=float)  # (T,4)
        if leader_hist.shape != (T, 4):
            raise ValueError(f"leader.history must have shape (T,4), got {leader_hist.shape}")
    else:
        x0 = _as_state4(leader).flatten()  # (4,)
        leader_hist = np.tile(x0, (T, 1))  # (T,4)

    N = len(agents)
    dim = 2 if use_pos_only else 4
    delta = np.zeros((T, N, dim), dtype=float)

    for i, ag in enumerate(agents):
        x_hist = np.array(ag.history, dtype=float)  # (T,4)
        if x_hist.shape != (T, 4):
            raise ValueError(f"agent.history must have shape (T,4), got {x_hist.shape}")

        f = np.array(ag.f, dtype=float).reshape(4, 1).flatten()  # (4,)
        d = x_hist - f - leader_hist  # (T,4)

        if use_pos_only:
            d = d[:, :2]  # (T,2)

        delta[:, i, :] = d

    return delta


def compute_delta_norms(agents, leader, use_pos_only=False):
    """
    Return per-agent error norms over time:

        norms[t,i] = ||delta_i(t)||_2

    Returns
    -------
    norms : np.ndarray shape (T, N)
    """
    delta = compute_delta(agents, leader, use_pos_only=use_pos_only)  # (T,N,dim)
    norms = np.linalg.norm(delta, axis=2)  # (T,N)
    return norms


def formation_error_mean(norms):
    """Mean over agents: shape (T,)"""
    norms = np.asarray(norms, dtype=float)
    return np.mean(norms, axis=1)


def formation_error_max(norms):
    """Max over agents: shape (T,)"""
    norms = np.asarray(norms, dtype=float)
    return np.max(norms, axis=1)


def formation_error_rms(norms):
    """RMS over agents: shape (T,)"""
    norms = np.asarray(norms, dtype=float)
    return np.sqrt(np.mean(norms**2, axis=1))


class ErrorTracker:
    """
    Track element-wise absolute tracking error (per time step):
        e_abs = |x_i - f_i - x0|

    Stores:
    - e_abs_hist: (T, 4N)
    - mean(|e|): (T,)
    - max(|e|):  (T,)
    """

    def __init__(self):
        self.e_abs_hist = []  # list of (4N,) arrays
        self.e_abs_mean = []  # list of floats
        self.e_abs_max = []   # list of floats

    def step(self, agents, leader_state):
        """
        Compute and store absolute error for current time step.

        leader_state can be:
        - Agent (has .state)
        - array-like (4,) or (4,1)
        """
        x0 = _as_state4(leader_state)  # (4,1)

        # Stack |x_i - f_i - x0| for all followers -> (4N,1)
        E_abs = np.vstack([np.abs(ag.state - ag.f - x0) for ag in agents])  # (4N,1)

        e_flat = E_abs.flatten()  # (4N,)
        self.e_abs_hist.append(e_flat)
        self.e_abs_mean.append(float(np.mean(e_flat)))
        self.e_abs_max.append(float(np.max(e_flat)))

    def get_history(self):
        """Return abs-error history as array (T, 4N)."""
        return np.array(self.e_abs_hist, dtype=float)

    def get_mean_history(self):
        """Return mean(|e|) per step as array (T,)."""
        return np.array(self.e_abs_mean, dtype=float)

    def get_max_history(self):
        """Return max(|e|) per step as array (T,)."""
        return np.array(self.e_abs_max, dtype=float)