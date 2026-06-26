import numpy as np

from .constraints import build_centralized_constraints
from .qp_solver import solve_cbf_qp

class CentralizedCBF():
    def __init__(self, gamma, safety_dis = 0.5):
        
        self.d_min = safety_dis
        self.gamma = gamma
        
    def compute_safe_control(self, all_agents, topology, u_nom, obstacles=None):

        N = len(all_agents)

        # QP cost: minimize ||u - u_nom||^2
        P = 2.0 * np.eye(2 * N)
        q = -2.0 * u_nom.flatten()

        G, h = build_centralized_constraints(
            all_agents=all_agents,
            topology=topology,
            d_min=self.d_min,
            gamma=self.gamma,
            obstacles=obstacles,
        )

        u_all_safe = solve_cbf_qp(P, q, G, h, solver="quadprog")

        if u_all_safe is not None:
            return u_all_safe.reshape(N, 2)

        else:
            print("Warning: QP infeasible. Returning zero controls.")
            return np.zeros((N, 2))
                
