import numpy as np
from qpsolvers import solve_qp

class DecentralizedCBF():
    def __init__(self, gamma, k = 1, safety_dis=0.5):
        """
        Initialize the Decentralized Control Barrier Function filter.
        :param gamma: Gain for the CBF constraint (higher = more aggressive).
        :param safety_dis: Minimum allowed distance (d_min) between agents.
        """
        self.d_min = safety_dis
        self.gamma = gamma
        self.p = 2*k + 1
        
    def compute_safe_control(self, agent_i, agent_id, all_agents, neighbor_list, u_nom_i):
        """
        Computes a safe control input for a single agent by solving a local QP.
        """
        n_vars = 3 # ux, uy, delta
        K = 1e5
        
        eps = 1e-6 # Small constant to prevent division by zero or overflows.
        
        # Objective: Minimize ||u_i - u_nom_i||^2
        # Standard QP form: (1/2)u^T P u + q^T u -> P=2I, q=-2*u_nom.
        P = 2.0 * np.eye(n_vars)
        P[0:2, 0:2] = 2.0 * np.eye(2)
        P[2, 2] = 2.0 * K
        
        q = np.zeros(n_vars)
        q[0:2] = -2.0 * u_nom_i.flatten()
        
        G_list = []
        h_list = []
        
        # 1. Collision Avoidance Constraints with Neighbors
        for j in neighbor_list:
            if j == agent_id:
                continue # Skip self-comparison
                
            agent_j = all_agents[j - 1]
            
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
            
            # Barrier dynamics components
            term_gamma = self.gamma * (h_ij**self.p) * dist
            term_projection = ((dv.T @ dp)**2) / (dist_sq + eps)
            term_v_norm = np.linalg.norm(dv)**2
            # Add eps to term_safe_v to prevent overflow/inf when at the safety boundary.
            term_accel = ((agent_i.alpha + agent_j.alpha) * (dv.T @ dp)) / (term_safe_v + eps)
            
            # pridicted_agent_j_acc = np.dot(dp,agent_j.alpha)
            
            # Total bound b_ij
            b_ij = term_gamma - term_projection + term_v_norm + term_accel
            
            # Distributed Responsibility: Share the burden based on max acceleration (alpha).
            distributed_term = agent_i.alpha / (agent_i.alpha + agent_j.alpha)
            val_b = float(np.asarray(b_ij * distributed_term).item())
            
            # Stability Check: Replace invalid numbers with a large negative value to force safety.
            if np.isinf(val_b) or np.isnan(val_b):
                val_b = -1e3 
            h_list.append(float(val_b))
            
            # G_ij * u - delta <= val_b
            G_list.append(np.array([-float(dp[0, 0]), -float(dp[1, 0]), -1.0]))
        
            # Inside the loop for a specific pair of agents
            # print(f"Dist: {dist:.2f} | h_ij: {h_ij:.2f} | b_ij: {val_b:.2f}")
        # Constraint: delta >= 0
        G_list.append([0, 0, -1.0])
        h_list.append(0.0)    
        
        # 2. Physical Limits (Box Constraints): |ui_x| <= alpha, |ui_y| <= alpha
        alpha = agent_i.alpha
        G_list.extend([[1,0,0], [-1,0,0], [0,1,0], [0,-1,0]])
        h_list.extend([alpha, alpha, alpha, alpha])

        # Convert lists to numpy arrays for the solver
        G = np.array(G_list)
        h = np.array(h_list).flatten()
        
        # 3. Solve the Local Quadratic Program
        sol = solve_qp(P, q, G, h, solver="quadprog")
        
        if sol is not None:
            u_safe = sol[0:2]
            slack_val = sol[2]
            if slack_val > 0.1:
                print(f"Warning: Safety violated! Slack: {slack_val:.4f}")
            return u_safe.reshape(2, 1)
    
        return np.zeros((2, 1))
    
    
    def compute_relax_safe_control(self, agent_i, agent_id, all_agents, neighbor_list, u_nom_i):
        """
        Computes a safe control input for a single agent by solving a local QP.
        """
        n_neighbor = len(neighbor_list)
        n_vars = 2 + n_neighbor # ux, uy, K
        cK = 1e6
        
        eps = 1e-6 # Small constant to prevent division by zero or overflows.
        
        P = 2.0 * np.eye(n_vars)
        
        q = np.zeros(n_vars)
        q[0:2] = -2.0 * u_nom_i.flatten()
        q[2:]  = -2.0 * np.sqrt(cK)
        
        G_list = []
        h_list = []
        
        # 1. Collision Avoidance Constraints with Neighbors
        for k, j in enumerate(neighbor_list):
            if j == agent_id:
                continue # Skip self-comparison
                
            agent_j = all_agents[j - 1]
            
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
            
            # # Constraint Matrix G: In decentralized mode, we only control agent i.
            # # The derivative of the barrier function leads to -dp^T * u_i.
            # G_list.append(-dp.flatten())
            
            # Barrier dynamics components
            term_gamma = self.gamma * (h_ij**self.p) * dist
            term_projection = ((dv.T @ dp)**2) / (dist_sq + eps)
            term_v_norm = np.linalg.norm(dv)**2
            # Add eps to term_safe_v to prevent overflow/inf when at the safety boundary.
            term_accel = ((agent_i.alpha + agent_j.alpha) * (dv.T @ dp)) / (term_safe_v + eps)
            
            # Total bound b_ij
            b_ij_bar = - term_projection + term_v_norm + term_accel
            
            # Distributed Responsibility: Share the burden based on max acceleration (alpha).
            distributed_term = agent_i.alpha / (agent_i.alpha + agent_j.alpha)
            val_b = b_ij_bar * distributed_term
            
            # Stability Check: Replace invalid numbers with a large negative value to force safety.
            if np.isinf(val_b) or np.isnan(val_b):
                val_b = -1e3 
            h_list.append(float(val_b.item()))
            
            # G_ij * u - delta <= val_b
            G_row = np.zeros(n_vars)
            G_row[0:2] = [-dp[0,0], -dp[1,0]]
            G_row[2 + k] = - (agent_i.alpha * term_gamma.item()/((agent_i.alpha + agent_j.alpha)*np.sqrt(cK)))

            G_list.append(G_row)
            # Inside the loop for a specific pair of agents
            # print(f"Dist: {dist:.2f} | h_ij: {h_ij:.2f} | b_ij: {val_b:.2f}")
            
            # Constraint: krj >= 1
            row_limit_krj = np.zeros(n_vars)
            row_limit_krj[2 + k] = -1.0
            G_list.append(row_limit_krj)
            h_list.append(-1.0)
            
        # 2. Physical Limits (Box Constraints): |ui_x| <= alpha, |ui_y| <= alpha
        alpha = agent_i.alpha
        
        # uix <= alpha_i
        row_limit_pos_x = np.zeros(n_vars); row_limit_pos_x[0] = 1
        # -uix <= alpha_i
        row_limit_neg_x = np.zeros(n_vars); row_limit_neg_x[0] = -1
        # uiy <= alpha_i
        row_limit_pos_y = np.zeros(n_vars); row_limit_pos_y[1] = 1
        # -uiy <= alpha_i
        row_limit_neg_y = np.zeros(n_vars); row_limit_neg_y[1] = -1
        
        G_list.extend([row_limit_pos_x, row_limit_neg_x, row_limit_pos_y, row_limit_neg_y])
        h_list.extend([alpha, alpha, alpha, alpha])
            
        # Convert lists to numpy arrays for the solver
        G = np.array(G_list)
        h = np.array(h_list).flatten()
        
        # 3. Solve the Local Quadratic Program
        sol = solve_qp(P, q, G, h, solver="quadprog")
        
        if sol is not None:
            u_safe = sol[0:2]
            # print(sol[3] / np.sqrt(cK))
            return u_safe.reshape(2, 1)

    
        return np.zeros((2, 1))