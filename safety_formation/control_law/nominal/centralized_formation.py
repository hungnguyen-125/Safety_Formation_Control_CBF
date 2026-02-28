import numpy as np
from ..base_controller import BaseController

class CentralizedFormationControl(BaseController):
    """
    Centralized formation controller using the Kronecker product and the
    augmented Laplacian matrix H.

    Control law:
        U_all = (I ⊗ K) * z + (I ⊗ K') * f 
        U_all = -(H ⊗ K) * (X_all - F_all - X_leader_all) + (I ⊗ K') * f 

    where:
    - H is the augmented Laplacian (N x N)
    - K is the local feedback gain (2 x 4)
    - K'is the formation compensation matrix to be designed (2 x 4)
    - X_all stacks follower states (4N x 1)
    - F_all stacks formation offsets (4N x 1)
    - X_leader_all repeats the leader state for each follower (4N x 1)
    """

    def __init__(self, K_matrix, K_prime_matrix = 0, name="CentralizedFormation"):
        super().__init__(name=name)
        self.K = np.array(K_matrix, dtype=float)
        self.K_prime = np.array(K_prime_matrix, dtype=float)

        # Safety checks on dimensions (expected K: 2x4 for double integrator in 2D)
        if self.K.shape != (2, 4):
            raise ValueError(f"K must have shape (2,4), got {self.K.shape}")
        if self.K_prime.shape != (2, 4):
            raise ValueError(f"K_prime must have shape (2,4), got {self.K_prime.shape}")

    def compute_nominal(self, all_agents, topology, leader_state=None):
        """
        Compute nominal control inputs for all followers simultaneously (centralized).

        Parameters
        ----------
        all_agents : list[Agent]
            List of follower agents (IDs 1..N).
        topology : Topology
            Topology object providing the augmented Laplacian H.
        leader_state : np.ndarray, optional
            Leader state vector of shape (4,1). If None, uses zeros.

        Returns
        -------
        U : np.ndarray
            Control matrix of shape (N, 2). Row i corresponds to follower i+1 (ID = i+1).
        """
        n = len(all_agents)      # Number of followers (N)
        dim_x = 4                # State dimension per agent: [x, y, vx, vy]
        dim_u = 2                # Input dimension per agent: [ax, ay]

        # 1) Get augmented Laplacian H from topology (N x N)
        H = topology.get_augmented_laplacian()

        if H.shape != (n, n):
            raise ValueError(f"H must have shape ({n},{n}), got {H.shape}")

        # 2) Build stacked state vector X_all (4N x 1) and formation vector F_all (4N x 1)
        X_all = np.vstack([agent.state for agent in all_agents])
        F_all = np.vstack([agent.f for agent in all_agents])

        if X_all.shape != (dim_x * n, 1):
            raise ValueError(f"X_all must have shape ({dim_x*n},1), got {X_all.shape}")
        if F_all.shape != (dim_x * n, 1):
            raise ValueError(f"F_all must have shape ({dim_x*n},1), got {F_all.shape}")

        # 3) Build stacked leader vector X_L_all (4N x 1)
        if leader_state is None:
            leader_state = np.zeros((dim_x, 1), dtype=float)
        else:
            leader_state = np.array(leader_state, dtype=float).reshape(dim_x, 1)

        X_L_all = np.tile(leader_state, (n, 1))

        # 4) Global error vector: E = X_all - F_all - X_L_all
        error_all = X_all - F_all - X_L_all

        # 5) Apply control law: U_all = -(H ⊗ K) * E + (I ⊗ K') * f
        H_kron_K = np.kron(H, self.K)    # shape: (2N x 4N)
        I_kron_K_prime = np.kron(np.eye(n), self.K_prime)    # shape: (2N x 4N)
        
        U_all = -H_kron_K @ error_all + I_kron_K_prime @ F_all     # shape: (2N x 1)

        # Return as (N x 2): each row is [ax, ay] for follower i
        return U_all.reshape(n, dim_u)