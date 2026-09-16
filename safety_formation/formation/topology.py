import numpy as np

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
        return bool(self.D_lead[agent_id - 1, agent_id - 1] > 0)
    
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
        from safety_formation.visualization import plot_topology

        return plot_topology(
            self,
            leader_pos=leader_pos,
            radius=radius,
            ax=ax,
            seed=seed,
        )


# from scipy.spatial.distance import pdist, squareform

# class DynamicDiskTopology:
#     def __init__(self, num_agents, alpha_list, beta_list, gamma, Ds):
#         """
#         alpha_list: List of maximum braking accelerations for each robot [alpha_1, ..., alpha_N]
#         beta_list: List of maximum speeds for each robot [beta_1, ..., beta_N]
#         gamma: ZCBF function parameter (coefficient of h^3)
#         Ds: Minimum safety distance
#         """
#         self.n = num_agents
#         self.alphas = np.array(alpha_list)
#         self.betas = np.array(beta_list)
#         self.gamma = gamma
#         self.Ds = Ds
        
#         # Compute the shared formation parameters according to the paper
#         self.alpha_min = np.min(self.alphas)
#         self.alpha_max = np.max(self.alphas)
#         self.beta_max = np.max(self.betas)
        
#         # Compute each robot's neighborhood radius using equation (10)
#         self.sensing_radii = self._compute_neighborhood_radii()
        
#         # Initialize an empty matrix
#         self.adj_matrix = np.zeros((self.n, self.n))

#     def _compute_neighborhood_radii(self):
#         """Implement the formula for D_N^i in image_350946.png"""
#         radii = []
#         for i in range(self.n):
#             # Term inside the large parentheses in equation (10)
#             term_sqrt = np.cbrt((2 * (self.alphas[i] + self.alpha_max)) / self.gamma)
#             main_term = (term_sqrt + self.betas[i] + self.beta_max)**2
            
#             # Complete formula for D_N^i
#             Di_N = self.Ds + (1 / (2 * (self.alphas[i] + self.alpha_min))) * main_term
#             radii.append(Di_N)
#         return np.array(radii)

#     def update(self, positions):
#         """Update adj_matrix based on current positions and D_N^i"""
#         dist_matrix = squareform(pdist(positions))
#         new_adj = np.zeros((self.n, self.n))
        
#         for i in range(self.n):
#             # Robot i only 'sees' robot j if the distance <= D_N^i
#             neighbors = np.where((dist_matrix[i] <= self.sensing_radii[i]) & (dist_matrix[i] > 0))[0]
#             new_adj[i, neighbors] = 1.0
            
#         self.adj_matrix = new_adj
#         # Then recompute the Laplacian if needed for formation control
