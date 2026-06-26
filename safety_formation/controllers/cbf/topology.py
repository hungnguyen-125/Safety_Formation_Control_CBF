import numpy as np


class CBFTopology:
    """
    Dynamic neighborhood topology induced by the CBF neighborhood radius.

    Neighbor rule:
        j in N_i  <=>  ||p_i - p_j|| <= D_N^i ,  j != i

    where
        D_N^i = D_s + 1 / (2 (alpha_i + alpha_min)) *
                ( cbrt( 2 (alpha_i + alpha_max) / gamma ) + beta_i + beta_max )^2
    """

    def __init__(self, agent_list, d_min, gamma):
        self.agent_list = agent_list
        self.n = len(agent_list)
        self.d_min = float(d_min)
        self.gamma = float(gamma)

        self.alpha_list = np.array([ag.alpha for ag in agent_list], dtype=float)
        self.beta_list = np.array([ag.beta for ag in agent_list], dtype=float)

        self.alpha_min = float(np.min(self.alpha_list))
        self.alpha_max = float(np.max(self.alpha_list))
        self.beta_max = float(np.max(self.beta_list))

        self.neighborhood_radii = self._compute_neighborhood_radii()
        self.adj_matrix = np.zeros((self.n, self.n), dtype=float)
        self.update_topology()

    def _compute_neighborhood_radii(self):
        radii = np.zeros(self.n, dtype=float)

        for i, ag in enumerate(self.agent_list):
            alpha_i = float(ag.alpha)
            beta_i = float(ag.beta)

            cubic_term = np.cbrt(2.0 * (alpha_i + self.alpha_max) / self.gamma )
            outer_term = (cubic_term + beta_i + self.beta_max) ** 2
            denom = 2.0 * (alpha_i + self.alpha_min)

            radii[i] = self.d_min + outer_term / denom

        return radii

    def update_topology(self):
        """
        Recompute adjacency based on current agent positions.
        Directed graph:
            adj_matrix[i, j] = 1 if j is in i's neighborhood.
        """
        self.adj_matrix = np.zeros((self.n, self.n), dtype=float)

        positions = [ag.pos.flatten() for ag in self.agent_list]

        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue

                dist_ij = np.linalg.norm(positions[i] - positions[j])
                if dist_ij <= self.neighborhood_radii[i]:
                    self.adj_matrix[i, j] = 1.0

        return self.adj_matrix

    def get_neighbors(self, agent_id):
        """
        Return neighbors of agent_id using 1-based agent IDs.
        """
        idx = agent_id - 1
        return np.where(self.adj_matrix[idx] > 0)[0] + 1

    def get_neighbors_by_index(self, idx):
        """
        Return neighbors using 0-based indexing.
        """
        return np.where(self.adj_matrix[idx] > 0)[0]

    def get_radius(self, agent_id):
        """
        Return D_N^i for 1-based agent ID.
        """
        return self.neighborhood_radii[agent_id - 1]

    def __repr__(self):
        return (
            f"CBFTopology(n={self.n}, d_min={self.d_min}, gamma={self.gamma}, "
            f"alpha_min={self.alpha_min}, alpha_max={self.alpha_max}, beta_max={self.beta_max})"
        )

    # ---------- visualization ----------
    def plot(self, ax=None, show_radius=False):
        """
        Plot current dynamic topology using actual agent positions.
        Directed edges are shown as undirected for simplicity.
        """
        from safety_formation.visualization import plot_cbf_topology

        return plot_cbf_topology(self, ax=ax, show_radius=show_radius)
