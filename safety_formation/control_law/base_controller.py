import numpy as np

class BaseController:
    def __init__(self, name="BaseController"):
        self.name = name

    def compute_nominal(self, agent_id, all_agents, topology):
        """
        Must be overridden in subclasses.

        Parameters
        ----------
        agent_id : int
            Follower ID in {1..N}.
        all_agents : list[Agent]
            List of follower agents. Convention: all_agents[i] has id = i+1.
        topology : Topology
            Topology object.
        """
        raise NotImplementedError("Subclasses must implement compute_nominal method")

    def get_neighbor_errors(self, agent_id, all_agents, topology):
        """
        Shared helper to compute consensus/formation error term:

            sum_{j in N_i} a_ij * [ (x_j - f_j) - (x_i - f_i) ]

        Notes
        -----
        - agent_id and returned neighbor IDs are 1-based.
        - Internally, Python lists are 0-based, so we map: idx = id - 1.
        """
        i = agent_id - 1  # map follower ID (1..N) -> list index (0..N-1)
        current_agent = all_agents[i]

        error_i = current_agent.state - current_agent.f
        error_sum = np.zeros_like(current_agent.state)

        # neighbors are returned as IDs (1..N)
        neighbor_ids = topology.get_neighbors(agent_id)

        for nid in neighbor_ids:
            j = nid - 1  # neighbor ID -> list index
            error_j = all_agents[j].state - all_agents[j].f

            # adjacency weight uses 0-based indices
            a_ij = topology.adj_matrix[i, j]
            error_sum += a_ij * (error_j - error_i)

        return error_sum