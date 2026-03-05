import numpy as np
from qpsolvers import solve_qp

class DecentralizedCBF():
    def __init__(self, gamma, safety_dis=0.5):
        """
        Initialize the Decentralized Control Barrier Function filter.
        :param gamma: Gain for the CBF constraint (higher = more aggressive).
        :param safety_dis: Minimum allowed distance (d_min) between agents.
        """
        self.d_min = safety_dis
        self.gamma = gamma
        
    def compute_safe_control(self, agent_i, agent_id, all_agents, neighbor_list, u_nom_i):
        """
        Computes a safe control input for a single agent by solving a local QP.
        """
        # The control input is 2D (accel_x, accel_y).
        n = 2 
        eps = 1e-6 # Small constant to prevent division by zero or overflows.
        
        # Objective: Minimize ||u_i - u_nom_i||^2
        # Standard QP form: (1/2)u^T P u + q^T u -> P=2I, q=-2*u_nom.
        P = 2.0 * np.eye(n)
        q = -2.0 * u_nom_i.flatten()
        
        G_list = []
        h_list = []
        
        # 1. Collision Avoidance Constraints with Neighbors
        for j in neighbor_list:
            if j == agent_id:
                continue # Skip self-comparison
                
            agent_j = all_agents[j]
            
            # Relative position and velocity vectors
            dp = agent_i.pos - agent_j.pos
            dv = agent_i.vel - agent_j.vel
            dist = np.linalg.norm(dp)
            dist_sq = dist**2
            dist = max(dist, 0.001) # Avoid singularity at zero distance
            
            # Calculate the safety barrier term h_ij
            # term_safe_v is related to the braking capability.
            safe_val = max(2 * (agent_i.alpha + agent_j.alpha) * (dist - self.d_min), 0)
            term_safe_v = np.sqrt(safe_val)
            h_ij = term_safe_v + (dp.T @ dv) / dist
            
            # Constraint Matrix G: In decentralized mode, we only control agent i.
            # The derivative of the barrier function leads to -dp^T * u_i.
            G_list.append(-dp.flatten())
            
            # Barrier dynamics components
            term_gamma = self.gamma * (h_ij**3) * dist
            term_projection = ((dv.T @ dp)**2) / (dist_sq + eps)
            term_v_norm = np.linalg.norm(dv**2)
            # Add eps to term_safe_v to prevent overflow/inf when at the safety boundary.
            term_accel = ((agent_i.alpha + agent_j.alpha) * (dv.T @ dp)) / (term_safe_v + eps)
            
            # Total bound b_ij
            b_ij = term_gamma - term_projection + term_v_norm + term_accel
            
            # Distributed Responsibility: Share the burden based on max acceleration (alpha).
            distributed_term = agent_i.alpha / (agent_i.alpha + agent_j.alpha)
            val_b = b_ij * distributed_term
            
            # Stability Check: Replace invalid numbers with a large negative value to force safety.
            if np.isinf(val_b) or np.isnan(val_b):
                val_b = -1e3 
            h_list.append(float(val_b))
            
            # Inside the loop for a specific pair of agents
            # print(f"Dist: {dist:.2f} | h_ij: {h_ij:.2f} | b_ij: {val_b:.2f}")
            
        # 2. Physical Limits (Box Constraints): |ui_x| <= alpha, |ui_y| <= alpha
        alpha = agent_i.alpha
        G_list.extend([
            np.array([1, 0]),  # uix <= alpha
            np.array([-1, 0]), # -uix <= alpha
            np.array([0, 1]),  # uiy <= alpha
            np.array([0, -1])  # -uiy <= alpha
        ])
        h_list.extend([alpha, alpha, alpha, alpha])

        # Convert lists to numpy arrays for the solver
        G = np.array(G_list)
        h = np.array(h_list).flatten()
        
        # 3. Solve the Local Quadratic Program
        u_safe = solve_qp(P, q, G, h, solver="quadprog")
        
        if u_safe is not None:
            return u_safe.reshape(2, 1)
        else:
            # Fallback: If no safe solution is found, stop the agent.
            print(f"Agent {agent_id} QP Failed!")
            return np.zeros((2, 1))