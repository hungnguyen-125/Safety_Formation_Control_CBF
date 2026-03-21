import numpy as np
from qpsolvers import solve_qp

class CentralizedCBF():
    def __init__(self, gamma, safety_dis = 0.5):
        
        self.d_min = safety_dis
        self.gamma = gamma
        
    def compute_safe_control(self, all_agents, topology, u_nom):
        
        N = len(all_agents)
        
        # Define QP cost matrices: minimize 1/2 * u^T * P * u + q^T * u
        # This setup minimizes ||u - u_nom||^2
        P = 2.0 * np.eye(2 * N)
        q = -2.0 * u_nom.flatten()
        
        G_list = []
        h_list = []
        
        # Iterate through agent pairs to define safety constraints
        for i in range(N):
            for j in range(i+1, N):
                if topology.adj_matrix[i, j] > 0 or topology.adj_matrix[j, i] > 0:
                    
                    agent_i = all_agents[i]
                    agent_j = all_agents[j]

                    dp = agent_i.pos - agent_j.pos
                    dv = agent_i.vel - agent_j.vel

                    # Avoid division by zero if agents are at the same position
                    dist = max(np.linalg.norm(dp), 0.001)
                    
                    # Calculate safety barrier function components
                    term_safe_v = np.sqrt(2 * (agent_i.alpha + agent_j.alpha) * (dist - self.d_min))
                    h_ij = term_safe_v + (dp.T @ dv) / dist

                    # Define the constraint row: G * u <= h
                    row_G = np.zeros(2 * N)
                    row_G[2*i : 2*i+2] = -dp.flatten() # Components for agent i control (u_i)
                    row_G[2*j : 2*j+2] = dp.flatten()  # Components for agent j control (u_j)
                    
                    term_gamma = self.gamma * (h_ij**3) * dist
                    term_projection = ((dv.T @ dp) / dist)**2
                    term_accel = ((agent_i.alpha + agent_j.alpha) * (dv.T @ dp)) / term_safe_v
                    term_v_norm = np.linalg.norm(dv)**2
                    
                    b_ij = np.array(term_gamma - term_projection + term_v_norm + term_accel)
                    
                    G_list.append(row_G)
                    
                    # Extract single value from array/list structures
                    val_b = np.asarray(b_ij).item() 
                    
                    # Numerical stability: handle infinite or NaN values
                    if np.isinf(val_b) or np.isnan(val_b):
                        # Replace with a large finite negative number to keep the constraint strict
                        val_b = -1e6 
                    h_list.append(float(val_b))
                    
        # Define individual actuator limits for each agent
        for i in range(N):
            limit = all_agents[i].alpha
            # Constraint: u_ix <= alpha_i
            row_limit_pos_x = np.zeros(2 * N); row_limit_pos_x[2*i] = 1
            # Constraint: -u_ix <= alpha_i
            row_limit_neg_x = np.zeros(2 * N); row_limit_neg_x[2*i] = -1
            # Constraint: u_iy <= alpha_i
            row_limit_pos_y = np.zeros(2 * N); row_limit_pos_y[2*i+1] = 1
            # Constraint: -u_iy <= alpha_i
            row_limit_neg_y = np.zeros(2 * N); row_limit_neg_y[2*i+1] = -1
            
            G_list.extend([row_limit_pos_x, row_limit_neg_x, row_limit_pos_y, row_limit_neg_y])
            h_list.extend([limit, limit, limit, limit])

        # --- 4. Solve the Quadratic Programming (QP) problem ---
        G = np.array(G_list)
        h = np.array(h_list).flatten()
        
        u_all_safe = solve_qp(P, q, G, h, solver="quadprog")
        
        # Reshape the flat u_all_safe result (2N,) back into a list of u_i (N, 2)
        if u_all_safe is not None:
            return u_all_safe.reshape(N, 2)
        
        # u_fallback = []

        # for ag in all_agents:
        #     v = ag.vel.flatten()
        #     v_norm = np.linalg.norm(v)

        #     if v_norm > 1e-6:
        #         u = -ag.alpha * v / v_norm
        #     else:
        #         u = np.zeros(2)

        #     u_fallback.append(u)

        # return np.array(u_fallback)
        
        else: 
            return np.zeros((N,2))
            

