import numpy as np
from scipy.optimize import linprog

from .constraints import build_decentralized_constraints
from .qp_solver import solve_cbf_qp

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
        
        # Objective: Minimize ||u_i - u_nom_i||^2
        # Standard QP form: (1/2)u^T P u + q^T u -> P=2I, q=-2*u_nom.
        P = 2.0 * np.eye(n_vars)
        P[0:2, 0:2] = 2.0 * np.eye(2)
        P[2, 2] = 2.0 * K
        
        q = np.zeros(n_vars)
        q[0:2] = -2.0 * u_nom_i.flatten()
        
        G, h = build_decentralized_constraints(
            agent_i=agent_i,
            agent_id=agent_id,
            all_agents=all_agents,
            neighbor_list=neighbor_list,
            d_min=self.d_min,
            gamma=self.gamma,
            p=self.p,
        )
        
        # 3. Solve the Local Quadratic Program
        sol = solve_cbf_qp(P, q, G, h, solver="quadprog")
        
        if sol is not None:
            u_safe = sol[0:2]
            slack_val = sol[2]
            if slack_val > 0.1:
                print(f"Warning: Safety violated! Slack: {slack_val:.4f}")
            return u_safe.reshape(2, 1)

        # elif slack_val > 10 and np.linalg.norm(agent_i.vel) == 0:
        #     return np.zeros((2, 1))
        
        # elif slack_val > 10 and np.linalg.norm(agent_i.vel) != 0:
        #     return -agent_i.alpha* agent_i.vel/np.linalg.norm(agent_i.vel)     

    def compute_relax_safe_control( self, agent_i, agent_id, all_agents, neighbor_list, u_nom_i, deadlock=None, u_safe_i=None, debug = False):
        """
        Decision variables:
            x = [u_x, u_y, z_0, z_1, ..., z_{n_neighbor-1}]^T

        where z_k = sqrt(cK) * k_gamma_k

        So:
            k_gamma_k = 1   <=> z_k = sqrt(cK)
            k_gamma_k > 1   <=> z_k > sqrt(cK)   (relaxed barrier)
            0 < k_gamma_k < 1 <=> 0 < z_k < sqrt(cK)  (compressed barrier)
        """

        cK = 1e6
        eps = 1e-6
        sqrt_cK = np.sqrt(cK)
        cross_tol = 1e-9

        k_gamma_left = 1.5
        k_gamma_right = 0.5

        # Check neighbor
        valid_neighbors = [j for j in neighbor_list if j != agent_id]

        n_neighbor = len(valid_neighbors)
        n_vars = 2 + n_neighbor

        # QP cost: minimize ||u - u_nom||^2 + sum (z_k - sqrt(cK))^2
        P = 2.0 * np.eye(n_vars)

        q = np.zeros(n_vars)
        q[0:2] = -2.0 * np.asarray(u_nom_i).reshape(2,)
        q[2:] = -2.0 * sqrt_cK
        if debug:
            print("\n=== GENERAL QP INPUT DEBUG ===")
            print("received u_nom_i:",
                np.asarray(u_nom_i).reshape(-1))
            print("q[:2]:", q[:2])

        G_list = []
        h_list = []
        dp_list = []

        for k, j in enumerate(valid_neighbors):
            agent_j = all_agents[j - 1]

            dp = np.asarray(agent_i.pos - agent_j.pos, dtype=float).reshape(2, 1)
            dv = np.asarray(agent_i.vel - agent_j.vel, dtype=float).reshape(2, 1)

            dp_list.append(dp.copy())

            dist = max(np.linalg.norm(dp), 1e-3)
            dist_sq = dist ** 2
            alpha_sum = agent_i.alpha + agent_j.alpha

            # h_ij
            safe_val = max(2.0 * alpha_sum * (dist - self.d_min), 0.0)
            term_safe_v = np.sqrt(safe_val)

            dpdv = float((dp.T @ dv).item())
            h_ij = term_safe_v + dpdv / dist

            # gamma term
            term_gamma = float(self.gamma * (h_ij ** self.p) * dist)

            term_projection = float((dpdv ** 2) / (dist_sq + eps))
            term_v_norm = float(np.linalg.norm(dv) ** 2)
            term_accel = float(alpha_sum * dpdv / (term_safe_v + eps))

            b_ij_bar = -term_projection + term_v_norm + term_accel
            distributed_term = agent_i.alpha / max(alpha_sum, eps)
            val_b = float(b_ij_bar * distributed_term)

            if not np.isfinite(val_b):
                val_b = -1e3

            # Constraint:
            #   -dp^T u_i - coeff * z_k <= val_b
            # with z_k = sqrt(cK) * k_gamma_k
            G_row = np.zeros(n_vars)
            G_row[0:2] = [-float(dp[0, 0]), -float(dp[1, 0])]
            G_row[2 + k] = -float( agent_i.alpha * term_gamma / (max(alpha_sum, eps) * sqrt_cK))

            G_list.append(G_row)
            h_list.append(val_b)

        G = np.array(G_list, dtype=float) if G_list else None
        h = np.array(h_list, dtype=float) if h_list else None

        # Bounds
        lb = -np.inf * np.ones(n_vars)
        ub = np.inf * np.ones(n_vars)

        # control bounds
        lb[0:2] = -agent_i.alpha
        ub[0:2] = agent_i.alpha

        # baseline: all z_k >= sqrt(cK), i.e. k_gamma >= 1
        lb[2:] = sqrt_cK

        # Type 1 deadlock handling:
        # left side  -> relaxed     -> z_k > sqrt(cK)
        # right side -> compressed  -> 0 < z_k < sqrt(cK)
        if deadlock == "type1":
            print(
                    f"neighbor {j}: "
                    f"h_ij={h_ij:.6f}, "
                    f"term_gamma={term_gamma:.6e}, "
                    f"val_b={val_b:.6e}"
                    )
            if u_safe_i is None:
                raise ValueError(
                    "u_safe_i must be provided for Type 1 deadlock handling."
                )

            u_hat = np.asarray(u_nom_i, dtype=float).reshape(2,)

            print("\n" + "=" * 60)
            print("TYPE 1 DEADLOCK DEBUG")
            print("=" * 60)

            print("agent_id:", agent_id)
            print("u_nom:", u_hat)
            print("u_safe before:", np.asarray(u_safe_i).reshape(-1))
            print("||u_nom||:", np.linalg.norm(u_hat))
            print("||u_safe||:", np.linalg.norm(u_safe_i))

            print("\n--- Neighbors ---")
            print("valid_neighbors:", valid_neighbors)

            print("\n--- Bounds BEFORE modification ---")
            print("lb:", lb.copy())
            print("ub:", ub.copy())

            # If nominal direction is nearly zero, fallback to no asymmetry
            if np.linalg.norm(u_hat) > cross_tol:

                for k, j in enumerate(valid_neighbors):
                    idx = 2 + k
                    dpk = np.asarray(dp_list[k], dtype=float).reshape(2,)

                    # 2D cross product scalar: u_hat x dpk
                    cross_val = (
                        u_hat[0] * dpk[1]
                        - u_hat[1] * dpk[0]
                    )

                    print(f"\nNeighbor {j}")
                    print("dp:", dpk)
                    print("cross_val:", cross_val)

                    # If neighbor is on the right: compress
                    if cross_val < 0.0:
                        lb[idx] = 0.0
                        ub[idx] = k_gamma_right * sqrt_cK

                        print("side: RIGHT")
                        print("action: COMPRESS")
                        print(
                            f"k_gamma bound: "
                            f"[0, {k_gamma_right}]"
                        )

                    # If neighbor is on the left: relax
                    else:
                        lb[idx] = k_gamma_left * sqrt_cK

                        print("side: LEFT")
                        print("action: RELAX")
                        print(
                            f"k_gamma bound: "
                            f"[{k_gamma_left}, inf)"
                        )

            else:
                print("WARNING: ||u_nom|| too small.")
                print("No Type 1 asymmetry applied.")

            print("\n--- QP matrices ---")
            print("G:\n", G)
            print("h:\n", h)

            print("\n--- Bounds AFTER modification ---")
            print("lb:", lb)
            print("ub:", ub)

            # More readable k_gamma bounds
            print("\n--- k_gamma bounds ---")
            for k, j in enumerate(valid_neighbors):
                idx = 2 + k

                kg_lb = lb[idx] / sqrt_cK
                kg_ub = ub[idx] / sqrt_cK

                print(
                    f"neighbor {j}: "
                    f"k_gamma in [{kg_lb}, {kg_ub}]"
                )

        sol = solve_cbf_qp(
            P,
            q,
            G,
            h,
            lb=lb,
            ub=ub,
            solver="quadprog"
        )

        if sol is not None:
            u_sol = sol[0:2]
            z_sol = sol[2:]
            k_gamma_sol = z_sol / sqrt_cK

            if deadlock == "type1":
                print("\n--- QP SOLUTION ---")
                print("u_sol:", u_sol)
                print("||u_sol||:", np.linalg.norm(u_sol))
                print("z_sol:", z_sol)
                print("k_gamma_sol:", k_gamma_sol)

                print(
                    "delta_u from old safe control:",
                    u_sol - np.asarray(u_safe_i).reshape(2,)
                )

                if G is not None:
                    residual = G @ sol - h

                    print("\n--- Constraint residual Gx - h ---")
                    print(residual)

                    print(
                        "max constraint violation:",
                        np.max(residual)
                    )

                    print(
                        "active constraints approximately:",
                        np.where(
                            np.abs(residual) <= 1e-3
                        )[0]
                    )

                print("=" * 60)
                print()

            return u_sol.reshape(2, 1)


        # Fallback if QP solver fails
        vel = np.asarray(agent_i.vel, dtype=float).reshape(2,)
        vel_norm = np.linalg.norm(vel)

        if deadlock == "type1":
            print("\n!!! QP SOLVER FAILED - USING FALLBACK !!!")
            print("agent:", agent_id)
            print("velocity:", vel)
            print("||velocity||:", vel_norm)

        if vel_norm < eps:
            if deadlock == "type1":
                print("Fallback output: zero control")
            return np.zeros((2, 1))

        u_fallback = -agent_i.alpha * vel / vel_norm

        if deadlock == "type1":
            print("Fallback output:", u_fallback)

        return u_fallback.reshape(2, 1)


    def compute_constraint(self, agent_i, agent_id, all_agents, neighbor_list, u_i = None):
       
        eps = 1e-6

        A_list = []
        b_list = []

        for j in neighbor_list:
            if j == agent_id:
                continue

            agent_j = all_agents[j - 1]

            # relative quantities, shape (2,1)
            dp = agent_i.pos - agent_j.pos
            dv = agent_i.vel - agent_j.vel

            dist = np.linalg.norm(dp)
            dist = max(dist, 1e-3)
            dist_sq = dist ** 2

            alpha_sum = agent_i.alpha + agent_j.alpha

            # h_ij
            safe_val = max(2.0 * alpha_sum * (dist - self.d_min), 0.0)
            term_safe_v = np.sqrt(safe_val)

            dpdv = float((dp.T @ dv).item())
            proj = dpdv / dist
            h_ij = term_safe_v + proj

            # terms in distributed CBF inequality
            term_gamma = float(self.gamma * (h_ij ** self.p) * dist)
            term_projection = float((dpdv ** 2) / (dist_sq + eps))
            term_v_norm = float(np.linalg.norm(dv) ** 2)
            term_accel = float((alpha_sum * dpdv) / (term_safe_v + eps))

            b_ij = term_gamma - term_projection + term_v_norm + term_accel

            # distributed responsibility
            distributed_term = agent_i.alpha / alpha_sum

            val_b = float(b_ij * distributed_term)

            if not np.isfinite(val_b):
                val_b = -1e3

            # Row of A_i u_i <= b_i
            # From: -dp^T u_i <= distributed_term * b_ij
            A_row = np.array([
                -float(dp[0, 0]),
                -float(dp[1, 0])
            ], dtype=float)

            A_list.append(A_row)
            b_list.append(val_b)
        
        alpha = float(agent_i.alpha)

        A_list.extend([
            np.array([ 1.0,  0.0]),
            np.array([-1.0,  0.0]),
            np.array([ 0.0,  1.0]),
            np.array([ 0.0, -1.0]),
        ])
        b_list.extend([alpha, alpha, alpha, alpha])

        if len(A_list) == 0:
            A = np.zeros((0, 2), dtype=float)
            b = np.zeros((0,), dtype=float)
        else:
            A = np.array(A_list, dtype=float)
            b = np.array(b_list, dtype=float)

        if u_i is None or np.linalg.norm(u_i) > 1e-3:
            n_active_constraints = 0
        else:
            n_active_constraints = count_active_constraints(u_i, A, b)
        
        return A, b, n_active_constraints

def count_active_constraints(u, A, b, tol=1e-3):
    if u is None:
        return 0

    ui = np.asarray(u, dtype=float).reshape(-1)

    slack = b - A @ ui

    return int(np.sum((slack >= -tol) & (slack <= tol)))