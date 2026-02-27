import numpy as np
from ..base_controller import BaseController

class CentralizedFormationControl(BaseController):
    """
    Replicate tutorial style: Điều khiển đội hình tập trung sử dụng 
    Kronecker product và ma trận Augmented Laplacian (H).
    Công thức: U_all = -(H ⊗ K) * (X_all - F_all - X_leader_all)
    """
    def __init__(self, K_matrix, name="CentralizedFormation"):
        super().__init__(name=name)
        self.K = np.array(K_matrix, dtype=float)

    def compute_all_controls(self, all_agents, topology, leader_state=None):
        """
        Tính toán u cho toàn bộ hệ thống cùng lúc (Centralized).
        """
        n = len(all_agents)      # Số lượng agent (ví dụ: 7)
        dim_x = 4                # Số trạng thái mỗi agent [x, y, vx, vy]
        dim_u = 2                # Số đầu vào mỗi agent [ax, ay]

        # 1. Lấy ma trận H từ topology
        H = topology.get_augmented_laplacian() # Kích thước (n x n)

        # 2. Xây dựng vector trạng thái tổng hợp X_all (4n x 1)
        # và vector formation tổng hợp F_all (4n x 1)
        X_all = np.vstack([agent.state for agent in all_agents])
        F_all = np.vstack([agent.f for agent in all_agents])

        # 3. Xây dựng vector Leader tổng hợp (nếu có leader)
        if leader_state is None:
            leader_state = np.zeros((dim_x, 1))
        X_L_all = np.tile(leader_state, (n, 1))

        # 4. Tính toán sai số tổng thể: E = X - F - X_L
        error_all = X_all - F_all - X_L_all

        # 5. Áp dụng luật điều khiển: U = -(H ⊗ K) * E
        # np.kron là tích Kronecker để khớp kích thước ma trận
        H_kron_K = np.kron(H, self.K)
        U_all = -H_kron_K @ error_all

        # Trả về kết quả dưới dạng list hoặc array (n x 2) để nạp vào từng agent
        return U_all.reshape(n, dim_u)