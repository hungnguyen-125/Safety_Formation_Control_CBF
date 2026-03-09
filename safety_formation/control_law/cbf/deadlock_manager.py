import numpy as np
from scipy.optimize import linprog

class DeadlockManager:
    """
    Implements Algorithm 1: Decentralized Deadlock Detection and Resolution.
    Manages the state of an agent to identify if it's stuck and applies 
    consistent perturbations to break the symmetry.
    """
    def __init__(self, agent_id, kd=0.5, lock_threshold=5):
        self.agent_id = agent_id
        self.kd = kd                  # Perturbation factor for Type 2 (k_delta)
        self.lock_threshold = lock_threshold  # Minimum frames to confirm a deadlock
        self.counter = 0              # Counts consecutive frames where u_safe is zero
        self.flag_lock = False        # High-level state indicating a resolved deadlock

    def decentralized_lp(self, G, h, alpha):
        """
        Step 1: Solves a Linear Program to find the safety margin delta_LP.
        Objective: Maximize delta_LP (Check if the admissible control space is empty).
        """
        # Objective vector: [ux, uy, delta_lp] -> we want to maximize delta_lp
        # Minimizing -delta_lp is equivalent to maximizing delta_lp.
        c = np.array([0, 0, -1.0]) 
        
        # Constraint: G*u <= h + delta_lp * 1  => G*u - delta_lp * 1 <= h
        # We augment the G matrix with a column of -1s for the delta variable.
        A_ub = np.hstack([G, -np.ones((G.shape[0], 1))])
        b_ub = h
        
        # Physical box constraints: |ux| <= alpha, |uy| <= alpha
        # delta_lp is unconstrained (can be positive or negative).
        bounds = [(-alpha, alpha), (-alpha, alpha), (None, None)]
        
        # Solve using the HiGHS method for efficiency
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            return res.x[2] # Return the optimized delta_LP
        return 0.0

    def check_deadlock_condition(self, agent, u_nom, u_safe):
        """
        Step 2: Logic to trigger Flag_lock based on nominal vs actual control.
        Condition: Wish to move (u_nom != 0) but forced to stop (u_safe == 0, v == 0).
        """
        is_stuck = (np.linalg.norm(u_nom) > 1e-2 and 
                    np.linalg.norm(u_safe) < 1e-2 and 
                    np.linalg.norm(agent.vel) < 1e-2)
        
        if is_stuck:
            self.counter += 1
        else:
            self.counter = 0
            self.flag_lock = False
            
        # Confirm deadlock only after a certain duration to avoid noise
        if self.counter >= self.lock_threshold:
            self.flag_lock = True
            
        return self.flag_lock

    def resolve(self, agent, u_nom, u_safe, G, h):
        """
        Main execution of Algorithm 1: Detect -> Categorize -> Perturb.
        """
        # 1. Check if the agent is currently in a deadlock state
        if not self.check_deadlock_condition(agent, u_nom, u_safe):
            return u_safe, False, None # No deadlock, proceed normally

        # 2. Categorize the Deadlock Scenario (Lines 5-12)
        delta_lp = self.decentralized_lp(G, h, agent.alpha)
        
        if delta_lp < 0:
            # Type 1 or 2: Admissible control space exists but optimal u is zero
            # If u_safe is at a vertex of the feasible set, it is Type 1.
            # If u_safe is on an edge, it is Type 2.
            dtype = 2 # Default to Type 2 for simplicity
        else:
            # Type 3: No feasible control exists (Safety space is empty)
            dtype = 3

        # 3. Apply Perturbation Strategy (Switch-case Lines 13-20)
        u_final = u_safe
        
        if dtype == 1:
            # Type 1: Requires asymmetric barrier gains (k_gamma_left/right)
            return u_safe, True, "TYPE_1" 
            
        elif dtype == 2:
            # Type 2: Apply a 90-degree normal perturbation to the left of u_nom
            rot_90_left = np.array([[0, -1], [1, 0]])
            delta_perp = self.kd * (rot_90_left @ u_nom.flatten())
            # Resulting command is the sum of nominal and perturbation
            u_final = (u_nom.flatten() + delta_perp).reshape(2,1)
            return u_final, True, "TYPE_2"
            
        elif dtype == 3:
            # Type 3: No perturbation possible without violating safety
            u_final = np.zeros((2,1))
            return u_final, True, "TYPE_3"

        return u_final, True, None