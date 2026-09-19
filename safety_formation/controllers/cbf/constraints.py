import numpy as np


def build_centralized_constraints(all_agents, topology, d_min, gamma, obstacles=None):
    N = len(all_agents)
    G_list = []
    h_list = []

    for i in range(N):
        for j in range(i + 1, N):
            if topology.adj_matrix[i, j] > 0 or topology.adj_matrix[j, i] > 0:
                agent_i = all_agents[i]
                agent_j = all_agents[j]

                dp = agent_i.pos - agent_j.pos
                dv = agent_i.vel - agent_j.vel

                dist = max(np.linalg.norm(dp), 0.001)
                dist_margin = max(dist - d_min, 1e-6)

                term_safe_v = np.sqrt(
                    2 * (agent_i.alpha + agent_j.alpha) * dist_margin
                )

                h_ij = term_safe_v + (dp.T @ dv) / dist

                row_G = np.zeros(2 * N)
                row_G[2 * i: 2 * i + 2] = -dp.flatten()
                row_G[2 * j: 2 * j + 2] = dp.flatten()

                term_gamma = gamma * (h_ij ** 3) * dist
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

    if obstacles is not None:
        for i, agent_i in enumerate(all_agents):
            p_i = np.asarray(agent_i.pos, dtype=float).reshape(-1)[:2]
            v_i = np.asarray(agent_i.vel, dtype=float).reshape(-1)[:2]

            for obs in obstacles:
                p_o = np.asarray(obs["center"], dtype=float).reshape(-1)[:2]
                r_o = float(obs["radius"])
                d_obs_safe = r_o + d_min

                dp = p_i - p_o
                dv = v_i

                dist = max(np.linalg.norm(dp), 0.001)
                dist_margin = max(dist - d_obs_safe, 1e-6)

                term_safe_v = np.sqrt(2 * agent_i.alpha * dist_margin)
                h_io = term_safe_v + (dp.T @ dv) / dist

                row_G = np.zeros(2 * N)
                row_G[2 * i: 2 * i + 2] = -dp.flatten()

                term_gamma = gamma * (h_io ** 3) * dist
                term_projection = ((dv.T @ dp) / dist) ** 2
                term_accel = (agent_i.alpha * (dv.T @ dp)) / term_safe_v
                term_v_norm = np.linalg.norm(dv) ** 2

                b_io = term_gamma - term_projection + term_v_norm + term_accel
                val_b = np.asarray(b_io).item()

                if np.isinf(val_b) or np.isnan(val_b):
                    val_b = -1e6

                G_list.append(row_G)
                h_list.append(float(val_b))

    append_centralized_actuator_limits(G_list, h_list, all_agents)

    return np.array(G_list), np.array(h_list).flatten()


def build_decentralized_constraints(agent_i, agent_id, all_agents, neighbor_list, d_min, gamma, p):
    eps = 1e-6
    G_list = []
    h_list = []

    for j in neighbor_list:
        if j == agent_id:
            continue

        agent_j = all_agents[j - 1]

        dp = agent_i.pos - agent_j.pos
        dv = agent_i.vel - agent_j.vel
        dist = np.linalg.norm(dp)
        dist_sq = dist**2
        dist = max(dist, 0.001)

        safe_val = max(2 * (agent_i.alpha + agent_j.alpha) * (dist - d_min), 0)
        term_safe_v = np.sqrt(safe_val)
        h_ij = term_safe_v + (dp.T @ dv) / dist

        term_gamma = gamma * (h_ij**p) * dist
        term_projection = ((dv.T @ dp)**2) / (dist_sq + eps)
        term_v_norm = np.linalg.norm(dv)**2
        term_accel = ((agent_i.alpha + agent_j.alpha) * (dv.T @ dp)) / (
            term_safe_v + eps
        )

        b_ij = term_gamma - term_projection + term_v_norm + term_accel
        distributed_term = agent_i.alpha / (agent_i.alpha + agent_j.alpha)
        val_b = float(np.asarray(b_ij * distributed_term).item())

        if np.isinf(val_b) or np.isnan(val_b):
            val_b = -1e3

        h_list.append(float(val_b))
        G_list.append(np.array([-float(dp[0, 0]), -float(dp[1, 0]), -1.0]))

    G_list.append([0, 0, -1.0])
    h_list.append(0.0)

    alpha = agent_i.alpha
    G_list.extend([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
    h_list.extend([alpha, alpha, alpha, alpha])

    return np.array(G_list), np.array(h_list).flatten()


def build_centralized_hocbf_constraints(all_agents, topology, d_min, k1_hocbf, k2_hocbf, obstacles=None):
    N = len(all_agents)
    G_list = []
    h_list = []

    for i in range(N):
        for j in range(i + 1, N):
            if topology.adj_matrix[i, j] > 0 or topology.adj_matrix[j, i] > 0:
                agent_i = all_agents[i]
                agent_j = all_agents[j]

                dp = (agent_i.pos - agent_j.pos).flatten()
                dv = (agent_i.vel - agent_j.vel).flatten()

                h_ij = np.dot(dp, dp) - d_min**2
                h_dot_ij = 2.0 * np.dot(dp, dv)

                row_G = np.zeros(2 * N)
                row_G[2 * i: 2 * i + 2] = -2.0 * dp
                row_G[2 * j: 2 * j + 2] = 2.0 * dp

                term_v_norm = 2.0 * np.linalg.norm(dv) ** 2
                term_h_dot = (k1_hocbf + k2_hocbf) * h_dot_ij
                term_h = k1_hocbf * k2_hocbf * h_ij

                b_ij = term_v_norm + term_h_dot + term_h
                val_b = float(b_ij)

                if not np.isfinite(val_b):
                    val_b = -1e6

                G_list.append(row_G)
                h_list.append(val_b)

    if obstacles is not None:
        for i, agent_i in enumerate(all_agents):
            p_i = np.asarray(agent_i.pos, dtype=float).reshape(-1)[:2]
            v_i = np.asarray(agent_i.vel, dtype=float).reshape(-1)[:2]

            for obs in obstacles:
                p_o = np.asarray(obs["center"], dtype=float).reshape(-1)[:2]
                r_o = float(obs["radius"])

                d_obs_safe = r_o + d_min

                dp = p_i - p_o
                dv = v_i

                h_io = np.dot(dp, dp) - d_obs_safe**2
                h_dot_io = 2.0 * np.dot(dp, dv)

                row_G = np.zeros(2 * N)
                row_G[2 * i: 2 * i + 2] = -2.0 * dp

                term_v_norm = 2.0 * np.linalg.norm(dv) ** 2
                term_h_dot = (k1_hocbf + k2_hocbf) * h_dot_io
                term_h = k1_hocbf * k2_hocbf * h_io

                b_io = term_v_norm + term_h_dot + term_h
                val_b = float(b_io)

                if not np.isfinite(val_b):
                    val_b = -1e6

                G_list.append(row_G)
                h_list.append(val_b)

    append_centralized_actuator_limits(G_list, h_list, all_agents)

    return np.array(G_list), np.array(h_list).flatten()


def append_centralized_actuator_limits(G_list, h_list, all_agents):
    N = len(all_agents)

    for i, agent in enumerate(all_agents):
        limit = agent.alpha

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
            row_limit_neg_y,
        ])

        h_list.extend([limit, limit, limit, limit])
