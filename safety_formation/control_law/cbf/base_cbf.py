import numpy as np
from qpsolvers import solve_qp # Cần cài đặt: pip install qpsolvers

class BaseCBF:
    def __init__(self, nominal_controller, safety_dist=0.5):
        """
        nominal_controller: Một instance kế thừa từ BaseController.
        safety_dist: Khoảng cách an toàn tối thiểu giữa các agent.
        """
        self.nominal = nominal_controller
        self.d_min = safety_dist

    def compute_safe_control(self, agent_id, all_agents, topology):
        # 1. Lấy tín hiệu nominal trước
        u_nom = self.nominal.compute_nominal(agent_id, all_agents, topology)
        
        # 2. Thiết lập bài toán QP: 
        # mụn tiêu: min |u - u_nom|^2 
        # ràng buộc: h_dot >= -gamma * h(x)
        u_safe = self.solve_cbf_qp(u_nom, agent_id, all_agents)
        return u_safe

    def solve_cbf_qp(self, u_nom, agent_id, all_agents):
        """
        Đây là lõi của CBF. Tùy vào Centralized hay Distributed 
        mà cách thiết lập ma trận ràng buộc sẽ khác nhau.
        """
        # Định nghĩa các ma trận cho QP: min 0.5 * u^T P u + q^T u
        # P = eye(2), q = -u_nom
        P = np.eye(2).astype(float)
        q = -u_nom.flatten().astype(float)
        
        # Các class con (Centralized/Distributed) sẽ định nghĩa G và h_vec
        # G @ u <= h_vec
        G, h_vec = self.generate_constraints(agent_id, all_agents)
        
        if G is None: # Không có vật cản/nguy cơ va chạm
            return u_nom

        # Giải QP
        u_opt = solve_qp(P, q, G, h_vec, solver="cvxopt")
        
        if u_opt is None:
            print(f"Warning: QP Infeasible for agent {agent_id}, using u_nom")
            return u_nom
            
        return u_opt.reshape(-1, 1)

    def generate_constraints(self, agent_id, all_agents):
        # Hàm này sẽ được ghi đè bởi CentralizedCBF hoặc DistributedCBF
        raise NotImplementedError("Subclasses must implement generate_constraints")