import numpy as np
from qpsolvers import solve_qp
from scipy.optimize import linprog

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

        # elif slack_val > 10 and np.linalg.norm(agent_i.vel) == 0:
        #     return np.zeros((2, 1))
        
        # elif slack_val > 10 and np.linalg.norm(agent_i.vel) != 0:
        #     return -agent_i.alpha* agent_i.vel/np.linalg.norm(agent_i.vel)     

    def compute_relax_safe_control( self, agent_i, agent_id, all_agents, neighbor_list, u_nom_i, deadlock=None, u_hat_i=None):
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

        # Check neighbor
        valid_neighbors = [j for j in neighbor_list if j != agent_id]

        n_neighbor = len(valid_neighbors)
        n_vars = 2 + n_neighbor

        # QP cost: minimize ||u - u_nom||^2 + sum (z_k - sqrt(cK))^2
        P = 2.0 * np.eye(n_vars)

        q = np.zeros(n_vars)
        q[0:2] = -2.0 * np.asarray(u_nom_i).reshape(2,)
        q[2:] = -2.0 * sqrt_cK

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
        if deadlock is not None:
            if u_hat_i is None:
                raise ValueError("u_hat_i must be provided for Type 1 deadlock handling.")

            u_hat = np.asarray(u_hat_i, dtype=float).reshape(2,)

            # If nominal direction is nearly zero, fallback to no asymmetry
            if np.linalg.norm(u_hat) > cross_tol:
                for k, j in enumerate(valid_neighbors):
                    idx = 2 + k
                    dpk = np.asarray(dp_list[k], dtype=float).reshape(2,)

                    # 2D cross product scalar: u_hat x dpk
                    cross_val = u_hat[0] * dpk[1] - u_hat[1] * dpk[0]

                    # If neighbor is on the right: compress
                    if cross_val < 0.0:
                        lb[idx] = 0.0
                        ub[idx] = sqrt_cK

                    # If neighbor is on the left: relax
                    else:
                        lb[idx] = sqrt_cK

        sol = solve_qp(P, q, G, h, lb=lb, ub=ub, solver="quadprog")

        if sol is not None:
            return sol[0:2].reshape(2, 1)

        # fallback
        vel = np.asarray(agent_i.vel, dtype=float).reshape(2,)
        vel_norm = np.linalg.norm(vel)

        if vel_norm < eps:
            return np.zeros((2, 1))

        return (-agent_i.alpha * vel / vel_norm).reshape(2, 1)


    def compute_constraint(self, agent_i, agent_id, all_agents, neighbor_list, include_box=False):
        """
        Build local decentralized CBF constraints for agent i:

            A_i u_i <= b_i

        where each row corresponds to one neighbor j.

        Parameters
        ----------
        agent_i : Agent
            Current agent i.
        agent_id : int
            1-based ID of agent i.
        all_agents : list
            List of agents ordered by ID, so all_agents[j-1] is agent j.
        neighbor_list : iterable
            1-based neighbor IDs of agent i.
        include_box : bool
            If True, append actuator box constraints:
                [ 1, 0] u <= alpha
                [-1, 0] u <= alpha
                [ 0, 1] u <= alpha
                [ 0,-1] u <= alpha

        Returns
        -------
        A : np.ndarray, shape (m, 2)
            Constraint matrix.
        b : np.ndarray, shape (m,)
            Constraint vector.
        """
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

        # optional actuator box constraints
        if include_box:
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

        return A, b