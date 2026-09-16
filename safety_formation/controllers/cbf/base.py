import numpy as np

from .qp_solver import solve_cbf_qp

class BaseCBF:
    def __init__(self, nominal_controller, safety_dist=0.5):
        """
        nominal_controller: An instance of a BaseController subclass.
        safety_dist: Minimum safety distance between agents.
        """
        self.nominal = nominal_controller
        self.d_min = safety_dist

    def compute_safe_control(self, agent_id, all_agents, topology):
        # 1. Compute the nominal control input first
        u_nom = self.nominal.compute_nominal(agent_id, all_agents, topology)
        
        # 2. Set up the QP problem:
        # objective: min |u - u_nom|^2
        # constraint: h_dot >= -gamma * h(x)
        u_safe = self.solve_cbf_qp(u_nom, agent_id, all_agents)
        return u_safe

    def solve_cbf_qp(self, u_nom, agent_id, all_agents):
        """
        This is the core of the CBF. Centralized and distributed formulations
        use different constraint matrix constructions.
        """
        # Define the matrices for the QP: min 0.5 * u^T P u + q^T u
        # P = eye(2), q = -u_nom
        P = np.eye(2).astype(float)
        q = -u_nom.flatten().astype(float)
        
        # Subclasses (Centralized/Distributed) define G and h_vec
        # G @ u <= h_vec
        G, h_vec = self.generate_constraints(agent_id, all_agents)
        
        if G is None: # No obstacles or collision risk
            return u_nom

        # Solve the QP
        u_opt = solve_cbf_qp(P, q, G, h_vec, solver="cvxopt")
        
        if u_opt is None:
            print(f"Warning: QP Infeasible for agent {agent_id}, using u_nom")
            return u_nom
            
        return u_opt.reshape(-1, 1)

    def generate_constraints(self, agent_id, all_agents):
        # This method is overridden by CentralizedCBF or DistributedCBF
        raise NotImplementedError("Subclasses must implement generate_constraints")
