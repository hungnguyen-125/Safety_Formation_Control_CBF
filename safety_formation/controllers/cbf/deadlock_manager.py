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

    def is_deadlocked(self, u_safe_i, u_nom_i, v_i):
        """
        Deadlock definition from the paper:
        deadlock if u_i = 0, v_i = 0, and u_nom_i != 0
        """
        if u_safe_i is None or u_nom_i is None or v_i is None:
            return False

        u_safe_norm = np.linalg.norm(np.asarray(u_safe_i).reshape(-1))
        uhat_norm = np.linalg.norm(np.asarray(u_nom_i).reshape(-1))
        v_norm = np.linalg.norm(np.asarray(v_i).reshape(-1))

        return (u_safe_norm <= self.zero_tol) and (v_norm <= 1) and (uhat_norm > self.zero_tol)

    def compute_delta_lp(self, A, b):
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
            (None, None),  # u_x
            (None, None),  # u_y
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

    def classify_deadlock(self, u_i, u_nom_i, v_i, A, b, n_active_constraints):
        """
        Returns:
            None      -> not in deadlock
            "type1"   -> deadlock at vertex
            "type2"   -> deadlock on edge
            "type3"   -> infeasible / empty admissible set
        """
        if not self.is_deadlocked(u_i, u_nom_i, v_i):
            return None

        delta_lp = self.compute_delta_lp(A, b)

        # Following the paper logic:
        # feasible QP region if delta_lp <= 0
        # type 3 if delta_lp >= 0 (or numerically close)
        if delta_lp >= -self.lp_tol:
            return "type3"

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

    def resolve_deadlock(
        self,
        cbf_solver,
        agent_i,
        agent_id,
        all_agents,
        neighbor_list,
        u_nom_i,
        u_safe_i,
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

        d_type = self.classify_deadlock(
            u_i=u_safe_i,
            u_nom_i=u_nom_i,
            v_i=agent_i.vel,
            A=A,
            b=b,
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
            return u_new, "type1"
        
        elif d_type == "type2":
            # Placeholder:
            # later replace by your perpendicular perturbation solver
            # for now, return original control
            return u_safe_i, "type2"

        else:  # type3
            return u_safe_i, "type3"