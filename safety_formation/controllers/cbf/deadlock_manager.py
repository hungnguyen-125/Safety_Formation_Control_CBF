import numpy as np
from scipy.optimize import linprog


class DeadlockManager:
    def __init__(self, zero_tol=1e-3, lp_tol=1e-7):
        """
        zero_tol : tolerance for checking ||u||, ||v||, ||u_hat||
        lp_tol   : tolerance for deciding LP feasibility margin
        """
        self.zero_tol = zero_tol
        self.lp_tol = lp_tol

        # Persistent turning direction for each agent.
        #
        # +1 : use R_ccw @ u_nom
        # -1 : use -(R_ccw @ u_nom)
        #
        # Type 1 decides this sign, Type 2 reuses it.
        self.turn_sign = {}

    def is_deadlocked(self, u_safe_i, u_nom_i, v_i):
        if u_safe_i is None or u_nom_i is None or v_i is None:
            return False

        u_safe_norm = np.linalg.norm(
            np.asarray(u_safe_i).reshape(-1)
        )
        u_nom_norm = np.linalg.norm(
            np.asarray(u_nom_i).reshape(-1)
        )
        v_norm = np.linalg.norm(
            np.asarray(v_i).reshape(-1)
        )

        return (
            u_safe_norm <= self.zero_tol
            and v_norm <= self.zero_tol
            and u_nom_norm > self.zero_tol
        )

    def compute_delta_lp(self, A, b, alpha):
        """
        Solve:
            min delta
            s.t. A u <= b + delta * 1

        Equivalent LP variables: x = [u_x, u_y, delta]
        Constraints:
            A[:,0:2] @ u - 1*delta <= b
        """
        if A is None or len(A) == 0:
            return -np.inf

        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float).reshape(-1)

        n_constraints = A.shape[0]

        # variables: [u_x, u_y, delta]
        c = np.array([0.0, 0.0, 1.0], dtype=float)

        A_ub = np.hstack([A, -np.ones((n_constraints, 1), dtype=float)])
        b_ub = b.copy()

        bounds = [
            (-alpha, alpha),  # u_x
            (-alpha, alpha),  # u_y
            (None, None),  # delta
        ]

        res = linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method="highs"
        )

        if not res.success:
            return np.inf

        return float(res.x[2])

    def classify_deadlock(self, u_i, u_nom_i, v_i, delta_lp, n_active_constraints):
        """
        Returns:
            None      -> not in deadlock
            "type1"   -> deadlock at vertex
            "type2"   -> deadlock on edge
            "type3"   -> infeasible / empty admissible set
        """
        if not self.is_deadlocked(u_i, u_nom_i, v_i):
            return None
        
        # feasible QP region if delta_lp <= 0
        # type 3 if delta_lp >= 0 (or numerically close)
        if delta_lp > -self.lp_tol:
            return "type3"

        if abs(delta_lp) <= self.lp_tol:
            return "degenerate"

        # feasible and deadlocked:
        # vertex => type 1
        # edge   => type 2
        if n_active_constraints >= 2:
            return "type1"
        elif n_active_constraints == 1:
            return "type2"

        # fallback: if deadlocked but active-count is weird numerically,
        # treat as type2 rather than fail hard
        return "type2"

    def compute_type2_safe_control(
        self,
        cbf_solver,
        agent_i,
        agent_id,
        all_agents,
        neighbor_list,
        u_nom_i,
        k_delta=0.2,
        debug=False,
        ):
        u_nom = np.asarray(u_nom_i, dtype=float).reshape(2,)

        if np.linalg.norm(u_nom) < 1e-9:
            return np.zeros((2, 1))

        R_ccw = np.array([
            [0.0, -1.0],
            [1.0, 0.0]
        ])

        turn_sign = self.turn_sign.get(
            agent_id,
            1.0
        )

        delta_perp = turn_sign * k_delta * (R_ccw @ u_nom)

        u_nom_pert = u_nom + delta_perp

        if debug:
            print("\n=== TYPE 2 DEADLOCK DEBUG ===")
            print("agent:", agent_id)
            print("u_nom:", u_nom)
            print("||u_nom||:", np.linalg.norm(u_nom))
            print("delta_perp:", delta_perp)
            print("u_nom_pert:", u_nom_pert)

        u_safe = cbf_solver.compute_relax_safe_control(
            agent_i=agent_i,
            agent_id=agent_id,
            all_agents=all_agents,
            neighbor_list=neighbor_list,
            u_nom_i=u_nom_pert.reshape(2, 1),
            deadlock=None,
            u_safe_i=None,
            debug = True,
        )

        if debug:
            print("u_safe after type2:", u_safe.reshape(-1))
            print("||u_safe||:", np.linalg.norm(u_safe))

        return u_safe

    def resolve_deadlock(
        self,
        cbf_solver,
        agent_i,
        agent_id,
        all_agents,
        neighbor_list,
        u_nom_i,
        u_safe_i,
        debug_type2=False
    ):
        """
        Main wrapper.

        Returns:
            u_resolved, deadlock_type
        """
        A, b, n_active = cbf_solver.compute_constraint(
            agent_i=agent_i,
            agent_id=agent_id,
            all_agents=all_agents,
            neighbor_list=neighbor_list,
            u_i=u_safe_i
        )

        delta_lp = self.compute_delta_lp(
        A,
        b,
        alpha=agent_i.alpha
        )

        # if self.is_deadlocked(
        #     u_safe_i,
        #     u_nom_i,
        #     agent_i.vel
        # ):
        #     print("=== DEADLOCK CANDIDATE ===")
        #     print("agent:", agent_id)
        #     print("||u_safe||:", np.linalg.norm(u_safe_i))
        #     print("||u_nom||:", np.linalg.norm(u_nom_i))
        #     print("||v||:", np.linalg.norm(agent_i.vel))
        #     print("A.shape:", A.shape)
        #     print("n_active:", n_active)
        #     print("delta_lp:", delta_lp)
        #     print("A:\n", A)
        #     print("b:\n", b)
        #     print()

        d_type = self.classify_deadlock(
            u_i=u_safe_i,
            u_nom_i=u_nom_i,
            v_i=agent_i.vel,
            delta_lp = delta_lp,
            n_active_constraints=n_active
        )

        if d_type is None:
            return u_safe_i, None

        if d_type == "type1":
            u_new = cbf_solver.compute_relax_safe_control(
                agent_i=agent_i,
                agent_id=agent_id,
                all_agents=all_agents,
                neighbor_list=neighbor_list,
                deadlock="type1",
                u_nom_i=u_nom_i,
                u_safe_i = u_safe_i
            )

            R_ccw = np.array([
                [0.0, -1.0],
                [1.0,  0.0]
            ])

            u_nom = np.asarray( u_nom_i, dtype=float).reshape(2,)

            u_old = np.asarray( u_safe_i, dtype=float).reshape(2,)

            u_new_vec = np.asarray( u_new, dtype=float).reshape(2,)

            # Direction introduced by Type 1
            delta_u = ( u_new_vec - u_old)

            # Reference CCW direction
            perp_ccw = (R_ccw @ u_nom)

            dot_val = np.dot( delta_u, perp_ccw)

            if dot_val > 0:
                turn_sign = 1.0

            elif dot_val < 0:
                turn_sign = -1.0

            else:
                # Exact zero only in pathological/numerical case.
                # Preserve previous direction if available.
                turn_sign = self.turn_sign.get(
                    agent_id,
                    1.0
                )

            # Store it persistently
            self.turn_sign[
                agent_id
            ] = turn_sign

            self.turn_sign[agent_id] = turn_sign

            return u_new, "type1"
        
        elif d_type == "type2":
            u_new = self.compute_type2_safe_control(
            agent_i=agent_i,
            cbf_solver = cbf_solver,
            agent_id=agent_id,
            all_agents=all_agents,
            neighbor_list=neighbor_list,
            u_nom_i=u_nom_i,
            k_delta=0.7,
            debug=debug_type2
            )
            
            print(
                "TYPE2 CHANGE:",
                "old =", np.asarray(u_safe_i).reshape(-1),
                "new =", np.asarray(u_new).reshape(-1),
            )
            return u_new, "type2"

        else:  # type3
            return u_safe_i, "type3"