import numpy as np

class BaseController:
    def __init__(self, name="BaseController"):
        self.name = name

    def compute_nominal(self, agent_id, all_agents, topology):
        """
        Hàm này bắt buộc phải được viết lại ở các class con.
        Input:
            agent_id: ID của agent hiện tại.
            all_agents: Danh sách các đối tượng Agent (để lấy state).
            topology: Đối tượng Topology (để lấy ma trận adjacency/Laplacian).
        Output:
            u_nom: Vector điều khiển [ux, uy]^T.
        """
        raise NotImplementedError("Subclasses must implement compute_nominal method")

    def get_neighbor_errors(self, agent_id, all_agents, topology):
        """
        Hàm hỗ trợ dùng chung để tính toán (x_j - f_j) - (x_i - f_i)
        Dùng cho các thuật toán dạng đồng thuận (Consensus).
        """
        current_agent = all_agents[agent_id]
        error_sum = np.zeros_like(current_agent.state)
        
        # Giả sử topology.adj_matrix cho biết ai là hàng xóm
        neighbors = np.where(topology.adj_matrix[agent_id] > 0)[0]
        
        for j in neighbors:
            error_j = all_agents[j].state - all_agents[j].f
            error_i = current_agent.state - current_agent.f
            error_sum += topology.adj_matrix[agent_id, j] * (error_j - error_i)
            
        return error_sum