import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class Topology:
    """
    Manages the network topology between agents in a Multi-Agent System.
    Provides adjacency, Laplacian, and augmented Laplacian matrices.
    """

    def __init__(self, num_agents, adjacency_matrix=None, leader_access=None):
        """
        Parameters
        ----------
        num_agents : int
            Number of follower agents (excluding the leader).
        adjacency_matrix : array-like, optional
            NxN adjacency matrix describing follower-to-follower connections.
        leader_access : array-like, optional
            1xN vector where entry i > 0 indicates that follower i has access to the leader.
        """
        self.n = num_agents

        # 1. Adjacency matrix among followers
        if adjacency_matrix is not None:
            self.adj_matrix = np.array(adjacency_matrix, dtype=float)
        else:
            self.adj_matrix = np.zeros((self.n, self.n))

        # 2. Leader access matrix (diagonal matrix D_lead)
        if leader_access is not None:
            self.D_lead = np.diag(leader_access)
        else:
            self.D_lead = np.zeros((self.n, self.n))

        # Precompute Laplacian matrices
        self.laplacian_matrix = self._compute_laplacian()
        self.augmented_laplacian = self._compute_augmented_laplacian()

    def _compute_laplacian(self):
        """
        Compute the graph Laplacian:
        L = D - A
        where D is the degree matrix and A is the adjacency matrix.
        """
        degree_matrix = np.diag(np.sum(self.adj_matrix, axis=1))
        return degree_matrix - self.adj_matrix

    def _compute_augmented_laplacian(self):
        """
        Compute the augmented Laplacian:
        H = L + D_lead
        where D_lead is the leader access matrix.
        """
        return self.laplacian_matrix + self.D_lead

    def get_augmented_laplacian(self):
        """
        Return the augmented Laplacian matrix H.
        Used in formation control analysis (e.g., stability theorems).
        """
        # latter use for dynamics topology 
        return self.augmented_laplacian

    def get_neighbors(self, agent_id):
        """
        Return the indices of neighboring followers of a given agent.
        """
        return np.where(self.adj_matrix[agent_id - 1] > 0)[0] + 1

    def sees_leader(self, agent_id):
        """
        Check whether the given agent has direct access to the leader.
        """
        return self.D_lead[agent_id - 1, agent_id - 1] > 0

    def __repr__(self):
        return f"Topology(n={self.n}, Leader_Linked={np.any(self.D_lead)})"
    
    ################################## Visualization ###################################
    
    def plot(self, leader_pos="center", radius=3.0, ax=None, seed=2):
        """
        Plot the network topology with follower indices 1..N and (optional) leader node 0.

        - Followers are placed on a circle.
        - Leader (if present) is placed at the center or at the bottom.
        - Leader edges are drawn as dashed lines.

        Parameters
        ----------
        leader_pos : {"center", "bottom"}
            Where to place the leader node (only used if a leader exists).
        radius : float
            Radius of the follower circle.
        ax : matplotlib.axes.Axes, optional
            If provided, draw on this axis. Otherwise create a new figure.
        seed : int
            Reserved for future layouts (kept for API stability).
        """

        # Determine whether leader exists
        has_leader = np.any(self.D_lead)

        # --- Build follower graph (nodes 1..N) ---
        G = nx.from_numpy_array(self.adj_matrix)
        G = nx.relabel_nodes(G, {i: i + 1 for i in range(self.n)})

        # --- Add leader node 0 only if there is at least one leader link ---
        if has_leader:
            G.add_node(0)
            for i in range(self.n):
                if self.D_lead[i, i] > 0:
                    G.add_edge(0, i + 1)

        # --- Nice positions: followers on a circle ---
        pos = {}
        angles = np.linspace(0, 2 * np.pi, self.n, endpoint=False)
        for i, ang in enumerate(angles, start=1):
            pos[i] = (radius * np.cos(ang), radius * np.sin(ang))

        # Leader position (only if leader exists)
        if has_leader:
            if leader_pos == "center":
                pos[0] = (0.0, 0.0)
            elif leader_pos == "bottom":
                pos[0] = (0.0, -1.4 * radius)
            else:
                raise ValueError("leader_pos must be 'center' or 'bottom'")

        # --- Plot ---
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))

        node_colors = [
            "red" if (has_leader and n == 0) else "skyblue"
            for n in G.nodes()
        ]

        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=900,
            edgecolors="k",
            linewidths=1,
            ax=ax
        )
        nx.draw_networkx_labels(G, pos, font_size=12, ax=ax)

        # Split follower edges vs leader edges for styling
        follower_edges = [(u, v) for (u, v) in G.edges() if (u != 0 and v != 0)]
        nx.draw_networkx_edges(G, pos, edgelist=follower_edges, width=2, ax=ax)

        if has_leader:
            leader_edges = [(u, v) for (u, v) in G.edges() if (u == 0 or v == 0)]
            nx.draw_networkx_edges(G, pos, edgelist=leader_edges, width=2, style="dashed", ax=ax)
            ax.set_title("Topology (Leader=0, Followers=1..N)")
        else:
            ax.set_title("Topology (Followers only)")

        ax.axis("off")
        return ax    
