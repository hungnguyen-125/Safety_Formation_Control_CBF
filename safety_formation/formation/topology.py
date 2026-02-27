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
        return self.D_lead[agent_id - 1, agent_id - 1] > 0

    def __repr__(self):
        return f"Topology(n={self.n}, Leader_Linked={np.any(self.D_lead)})"