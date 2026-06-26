import numpy as np
from qpsolvers import solve_qp

class CentralizedCBF():
    def __init__(self, gamma, safety_dis = 0.5):
        
        self.d_min = safety_dis
        self.gamma = gamma
        
    def compute_safe_control(self, all_agents, topology, u_nom, obstacles=None):

        N = len(all_agents)

        # QP cost: minimize ||u - u_nom||^2
        P = 2.0 * np.eye(2 * N)
        q = -2.0 * u_nom.flatten()

        G_list = []
        h_list = []

        # ============================================================
        # 1. Agent-agent CBF constraints
        # ============================================================
        for i in range(N):
            for j in range(i + 1, N):

                if topology.adj_matrix[i, j] > 0 or topology.adj_matrix[j, i] > 0:

                    agent_i = all_agents[i]
                    agent_j = all_agents[j]

                    dp = agent_i.pos - agent_j.pos
                    dv = agent_i.vel - agent_j.vel

                    dist = max(np.linalg.norm(dp), 0.001)

                    # If already unsafe, avoid sqrt of negative number
                    dist_margin = max(dist - self.d_min, 1e-6)

                    term_safe_v = np.sqrt(
                        2 * (agent_i.alpha + agent_j.alpha) * dist_margin
                    )

                    h_ij = term_safe_v + (dp.T @ dv) / dist

                    row_G = np.zeros(2 * N)

                    row_G[2 * i: 2 * i + 2] = -dp.flatten()
                    row_G[2 * j: 2 * j + 2] = dp.flatten()

                    term_gamma = self.gamma * (h_ij ** 3) * dist
                    term_projection = ((dv.T @ dp) / dist) ** 2
                    term_accel = (
                        ((agent_i.alpha + agent_j.alpha) * (dv.T @ dp))
                        / term_safe_v
                    )
                    term_v_norm = np.linalg.norm(dv) ** 2

                    b_ij = term_gamma - term_projection + term_v_norm + term_accel

                    val_b = np.asarray(b_ij).item()

                    if np.isinf(val_b) or np.isnan(val_b):
                        val_b = -1e6

                    G_list.append(row_G)
                    h_list.append(float(val_b))

        # ============================================================
        # 2. Agent-obstacle CBF constraints
        # obstacles format:
        # obstacles = [
        #     {"center": np.array([0.0, 0.5]), "radius": 0.35},
        #     {"center": np.array([0.0, -0.5]), "radius": 0.35}
        # ]
        # ============================================================
        if obstacles is not None:

            for i, agent_i in enumerate(all_agents):

                p_i = np.asarray(agent_i.pos, dtype=float).reshape(-1)[:2]
                v_i = np.asarray(agent_i.vel, dtype=float).reshape(-1)[:2]

                for obs in obstacles:

                    p_o = np.asarray(obs["center"], dtype=float).reshape(-1)[:2]
                    r_o = float(obs["radius"])
                    
                    # Safe distance from obstacle center
                    d_obs_safe = r_o + self.d_min

                    dp = p_i - p_o
                    dv = v_i  # obstacle is static, so v_o = 0

                    dist = max(np.linalg.norm(dp), 0.001)

                    # Avoid sqrt of negative number if already inside unsafe zone
                    dist_margin = max(dist - d_obs_safe, 1e-6)

                    # Use obstacle as static object with alpha = agent_i.alpha
                    term_safe_v = np.sqrt(
                        2 * agent_i.alpha * dist_margin
                    )

                    h_io = term_safe_v + (dp.T @ dv) / dist

                    row_G = np.zeros(2 * N)

                    # Only u_i appears because obstacle has no control
                    row_G[2 * i: 2 * i + 2] = -dp.flatten()

                    term_gamma = self.gamma * (h_io ** 3) * dist
                    term_projection = ((dv.T @ dp) / dist) ** 2
                    term_accel = (agent_i.alpha * (dv.T @ dp)) / term_safe_v
                    term_v_norm = np.linalg.norm(dv) ** 2

                    b_io = term_gamma - term_projection + term_v_norm + term_accel

                    val_b = np.asarray(b_io).item()

                    if np.isinf(val_b) or np.isnan(val_b):
                        val_b = -1e6

                    G_list.append(row_G)
                    h_list.append(float(val_b))

        # ============================================================
        # 3. Actuator limits
        # ============================================================
        for i in range(N):

            limit = all_agents[i].alpha

            row_limit_pos_x = np.zeros(2 * N)
            row_limit_pos_x[2 * i] = 1

            row_limit_neg_x = np.zeros(2 * N)
            row_limit_neg_x[2 * i] = -1

            row_limit_pos_y = np.zeros(2 * N)
            row_limit_pos_y[2 * i + 1] = 1

            row_limit_neg_y = np.zeros(2 * N)
            row_limit_neg_y[2 * i + 1] = -1

            G_list.extend([
                row_limit_pos_x,
                row_limit_neg_x,
                row_limit_pos_y,
                row_limit_neg_y
            ])

            h_list.extend([limit, limit, limit, limit])

        # ============================================================
        # 4. Solve QP
        # ============================================================
        G = np.array(G_list)
        h = np.array(h_list).flatten()

        u_all_safe = solve_qp(P, q, G, h, solver="quadprog")

        if u_all_safe is not None:
            return u_all_safe.reshape(N, 2)

        else:
            print("Warning: QP infeasible. Returning zero controls.")
            return np.zeros((N, 2))
                

