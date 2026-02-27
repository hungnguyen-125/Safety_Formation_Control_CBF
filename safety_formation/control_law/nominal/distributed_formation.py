import numpy as np
from ..base_controller import BaseController

class DistributedFormationControl(BaseController):
    """
    Triển khai thuật toán Formation Control dựa trên Consensus Protocol (16).
    u_i = -K * [ sum_{j in N_i} a_ij * ((x_i - f_i) - (x_j - f_j)) + b_i * (x_i - f_i - x_L) ]
    """
    def __init__(self, K_matrix, name="DistributedFormationControl"):
        super().__init__(name=name)
        # K_matrix thường có kích thước (2, 4) cho hệ Double Integrator
        self.K = np.array(K_matrix, dtype=float)

    def compute_nominal(self, agent_id, all_agents, topology, leader_state=None):
        """
        Tính toán tín hiệu u cho một agent cụ thể.
        :param agent_id: ID của agent đang xét.
        :param all_agents: Danh sách toàn bộ các object Agent.
        :param topology: Object Topology chứa ma trận H (Augmented Laplacian).
        :param leader_state: Trạng thái leader x_L (mặc định là [0,0,0,0]^T nếu không truyền).
        """
        if leader_state is None:
            leader_state = np.zeros((4, 1))
        else:
            leader_state = np.array(leader_state).reshape(4, 1)

        current_agent = all_agents[agent_id]
        
        # 1. Tính sai số formation cục bộ của agent i: e_i = x_i - f_i - x_L
        # Lưu ý: Trong Protocol (16), sai số này được tính tương đối với Leader
        error_i = current_agent.state - current_agent.f - leader_state
        
        # 2. Tính tổng sai số với các hàng xóm (Consensus term)
        # sum_{j in neighbors} a_ij * ( (x_i - f_i) - (x_j - f_j) )
        neighbor_sum = np.zeros((4, 1))
        neighbors = topology.get_neighbors(agent_id)
        
        for j in neighbors:
            neighbor_agent = all_agents[j]
            diff_i = current_agent.state - current_agent.f
            diff_j = neighbor_agent.state - neighbor_agent.f
            
            # Trọng số liên kết a_ij từ ma trận Adjacency
            a_ij = topology.adj_matrix[agent_id, j]
            neighbor_sum += a_ij * (diff_i - diff_j)

        # 3. Tính thành phần liên kết với Leader: b_i * (x_i - f_i - x_L)
        # b_i lấy từ đường chéo của ma trận b_diag trong topology
        b_i = topology.b_diag[agent_id, agent_id]
        leader_term = b_i * error_i

        # 4. Tổng hợp tín hiệu điều khiển: u_i = -K * (neighbor_sum + leader_term)
        u_nominal = -self.K @ (neighbor_sum + leader_term)
        
        return u_nominal.reshape(2, 1)