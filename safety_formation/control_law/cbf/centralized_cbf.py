import numpy as np
from qpsolvers import solve_qp

class CentralizedCBF():
    def __init__(self, gamma, safety_dis = 0.5):
        
        self.d_min = safety_dis
        
        self.gamma = gamma
        
    def compute_safe_control(self, all_agents, topology, u_nom):
        
        N = len(all_agents)
        
        P = 2.0 *np.eye(2* N)
        q = -2.0* u_nom.flatten()
        
        G_list = []
        h_list = []
        
        for i in range(N):
            for j in range(i+1, N):
                if topology.adj_matrix[i, j] > 0:
                    agent_i = all_agents[i]
                    agent_j = all_agents[j]

                    dp = agent_i.pos - agent_j.pos
                    dv = agent_i.vel - agent_j.vel

                    dist = max(np.linalg.norm(dp), 0.001)
                    
                    term_safe_v = np.sqrt(2*(agent_i.alpha + agent_j.alpha)*(dist - self.d_min))
                    h_ij = term_safe_v + (dp.T@dv)/dist

                    row_G = np.zeros(2 * N)
                    row_G[2*i : 2*i+2] = -dp.flatten() # Phần cho u_i
                    row_G[2*j : 2*j+2] = dp.flatten()  # Phần cho u_j
                    
                    term_gamma = self.gamma * (h_ij**3) * dist
                    term_projection = ((dv.T@dp)/dist)**2
                    term_accel = ((agent_i.alpha + agent_j.alpha)*(dv.T@dp)) / term_safe_v
                    term_v_norm = np.linalg.norm(dv)**2
                    
                    b_ij = term_gamma - term_projection + term_v_norm + term_accel

                    G_list.append(row_G)
                    h_list.append(b_ij)
                    
        for i in range(N):
            limit = all_agents[i].alpha
            # uix <= alpha_i
            row_limit_pos_x = np.zeros(2 * N); row_limit_pos_x[2*i] = 1
            # -uix <= alpha_i
            row_limit_neg_x = np.zeros(2 * N); row_limit_neg_x[2*i] = -1
            # uiy <= alpha_i
            row_limit_pos_y = np.zeros(2 * N); row_limit_pos_y[2*i+1] = 1
            # -uiy <= alpha_i
            row_limit_neg_y = np.zeros(2 * N); row_limit_neg_y[2*i+1] = -1
            
            G_list.extend([row_limit_pos_x, row_limit_neg_x, row_limit_pos_y, row_limit_neg_y])
            h_list.extend([limit, limit, limit, limit])

        # --- 4. Giải bài toán QP ---
        G = np.array(G_list)
        h = np.array(h_list).flatten()
        
        u_all_safe = solve_qp(P, q, G, h, solver="quadprog")
        
        # Tách kết quả u_all_safe (2N,) thành danh sách các u_i (2,1)
        if u_all_safe is not None:
            return u_all_safe.reshape(N, 2)
        else:
            print("QP Solver failed to find a solution!")
            return None