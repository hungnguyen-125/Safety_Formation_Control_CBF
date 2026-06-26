from qpsolvers import solve_qp


def solve_cbf_qp(P, q, G=None, h=None, lb=None, ub=None, solver="quadprog"):
    """Solve a CBF quadratic program using the configured qpsolvers backend."""
    return solve_qp(P, q, G, h, lb=lb, ub=ub, solver=solver)
