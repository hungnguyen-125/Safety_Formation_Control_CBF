import numpy as np
from ..base import BaseController

class DistributedFormationControl(BaseController):
    """
    Implement formation control based on Consensus Protocol (16).
    u_i = -K * [ sum_{j in N_i} a_ij * ((x_i - f_i) - (x_j - f_j)) + b_i * (x_i - f_i - x_L) ]
    """
    def __init__(self, K_matrix, name="DistributedFormationControl"):
        super().__init__(name=name)
        # K_matrix typically has shape (2, 4) for a double integrator system
        self.K = np.array(K_matrix, dtype=float)

    def compute_nominal(self, agent_id, all_agents, topology, leader_state=None):
        """
        Compute the control input u for a specific agent.
        :param agent_id: ID of the current agent.
        :param all_agents: List of all Agent objects.
        :param topology: Topology object containing the augmented Laplacian matrix H.
        :param leader_state: Leader state x_L (defaults to [0,0,0,0]^T if omitted).
        """
        if leader_state is None:
            leader_state = np.zeros((4, 1))
        else:
            leader_state = np.array(leader_state).reshape(4, 1)

        current_agent = all_agents[agent_id]
        
        # 1. Compute the local formation error for agent i: e_i = x_i - f_i - x_L
        # Note: In Protocol (16), this error is computed relative to the leader
        error_i = current_agent.state - current_agent.f - leader_state
        
        # 2. Sum the errors relative to neighbors (consensus term)
        # sum_{j in neighbors} a_ij * ( (x_i - f_i) - (x_j - f_j) )
        neighbor_sum = np.zeros((4, 1))
        neighbors = topology.get_neighbors(agent_id)
        
        for j in neighbors:
            neighbor_agent = all_agents[j]
            diff_i = current_agent.state - current_agent.f
            diff_j = neighbor_agent.state - neighbor_agent.f
            
            # Edge weight a_ij from the adjacency matrix
            a_ij = topology.adj_matrix[agent_id, j]
            neighbor_sum += a_ij * (diff_i - diff_j)

        # 3. Compute the leader coupling term: b_i * (x_i - f_i - x_L)
        # b_i comes from the diagonal of the topology's b_diag matrix
        b_i = topology.b_diag[agent_id, agent_id]
        leader_term = b_i * error_i

        # 4. Combine the control terms: u_i = -K * (neighbor_sum + leader_term)
        u_nominal = -self.K @ (neighbor_sum + leader_term)
        
        return u_nominal.reshape(2, 1)
