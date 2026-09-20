# Safe Formation Control for Multi-Robot Systems

Multi-robot formation control must satisfy two competing objectives:

- track a desired formation relative to a leader or reference trajectory;
- maintain safety by avoiding inter-agent and obstacle collisions while respecting actuator constraints.

This repository studies a two-layer control architecture:

1. a **distributed formation controller** that generates the nominal control input;
2. a **CBF-QP safety filter** that minimally modifies the nominal input whenever safety constraints are at risk.

<p align="center">
  <img
    src="docs/figures/proposed-control-structure.png"
    alt="Proposed Control Structure"
    style="width: 100%; max-width: 800px; height: auto;"
  >
</p>

## Project Objectives

### 1. Distributed formation control

The first objective is to understand and implement a distributed state-feedback formation controller. The feedback gains \(K\) and \(K'\) are obtained through an LMI-based design and are used to drive the multi-robot system toward the desired formation using local information.

This controller serves as the **nominal controller** of the overall framework.

### 2. CBF-QP safety filtering

The second objective is to understand, implement, and compare several Control Barrier Function (CBF)-based safety filters:

- Centralized Zeroing CBF (ZCBF)
- Distributed ZCBF
- Relaxed Distributed CBF (RDCBF)
- Centralized High-Order CBF (HOCBF)

The safety filter solves a quadratic program (QP) that keeps the applied control input as close as possible to the nominal controller while enforcing safety constraints.

## Framework

The implemented framework considers:

- inter-agent collision avoidance;
- obstacle avoidance;
- actuator and velocity limits;
- centralized and distributed safety-filter architectures.

A common simulation environment is used to compare the different CBF formulations in terms of behavior, conservativeness, and computational characteristics under the same multi-robot scenarios.

## Problem Formulation

Each robot is modeled as a planar double-integrator system:

```text
p_i_dot = v_i
v_i_dot = u_i
```

where `p_i` is the robot position, `v_i` is the robot velocity, and `u_i` is the
acceleration control input.

The formation objective is to drive each follower robot toward a desired offset
from a leader or reference state:

```text
e_i = x_i - f_i - x_L
```

where `x_i = [p_x, p_y, v_x, v_y]^T`, `f_i` is the desired formation offset, and
`x_L` is the leader state.

The safety objective is to keep every relevant pair of robots separated by at
least a prescribed safety distance:

```text
||p_i - p_j|| >= d_min
```

For obstacle scenarios, the same idea is applied between each robot and circular
obstacles using an inflated obstacle radius:

```text
||p_i - p_obs|| >= r_obs + d_min
```

## CBF Safety Filter

The centralized CBF filter solves:

```text
minimize    ||u - u_nom||^2
subject to  G u <= h
```

where `u_nom` is the nominal formation-control input and `u` is the safe control
input returned by the filter.

The centralized constraint builder includes:

- Robot-robot safety constraints for connected agents in the topology graph.
- Robot-obstacle safety constraints for circular obstacles.
- Actuator bounds based on each agent acceleration limit `alpha`.

The repository also includes a centralized HOCBF implementation. HOCBF is useful
for double-integrator dynamics because the distance-based safety function has
relative degree two with respect to acceleration input.

The CBF and HOCBF implementations are validated in the demo notebooks and media
files, including:

- `demo/media/centralized_cbf_validation.mp4`
- `demo/media/centralized_hocbf_validation.mp4`
- `demo/media/centralized_cbf_formation_control.mp4`
- `demo/media/Formation_with_obstacle_and_disturbance.mp4`

## RDCBF Formulation

The relaxed decentralized CBF (RDCBF) formulation extends the local CBF filter
by giving each robot its own local QP. Instead of solving one global problem for
all robots, each robot computes a safe local control using only its neighboring
robots.

For robot `i`, the local QP minimizes the deviation from its nominal input:

```text
minimize    ||u_i - u_nom_i||^2
subject to  local CBF constraints with neighbors j in N_i
            actuator limits
```

The relaxed version introduces additional relaxation variables that modify the
effective CBF gain for each neighbor. This allows the controller to handle
geometrically difficult cases where a strict decentralized CBF may become too
conservative.

The current RDCBF implementation includes the idea of asymmetric relaxation:

- Increase the CBF gain for neighbors on one side.
- Compress or reduce the effective gain for neighbors on the other side.
- Use the nominal direction to decide which side should be relaxed or compressed.

This is intended to create a small preference in the local feasible set so that
robots can escape symmetric blocking configurations.

Deadlock resolution is still under development. The current idea is to detect
when the filtered control is nearly zero while the nominal controller still
wants to move, classify the deadlock geometry, and then perturb or relax the
local CBF constraints to recover motion. The implementation in
`safety_formation/controllers/cbf/deadlock_manager.py` should be considered
experimental and not yet a final validated contribution.

## Simulation Scenarios

The repository contains examples and validation assets for several scenarios:

- Basic double-integrator simulation with a custom control policy.
- Centralized CBF validation for robot collision avoidance.
- Centralized HOCBF validation for relative-degree-two safety constraints.
- Centralized formation control with CBF safety filtering.
- Decentralized CBF validation.
- Relaxed decentralized CBF validation.
- Formation control with obstacle avoidance and disturbance injection.

Generated videos are stored in `demo/media/`, and exploratory notebooks are in
`demo/notebooks/`.

## Results

The current results show that the CBF safety filter can modify nominal formation
commands while preserving collision avoidance constraints. In the centralized
CBF and HOCBF validation scenarios, robots are able to move toward their goals
or formation targets while maintaining the required minimum separation.

The obstacle and disturbance scenario demonstrates that the same safety-filter
structure can include circular obstacles and external control disturbances. The
filter still projects the nominal command back into the safe set when constraints
become active.

The decentralized and relaxed decentralized experiments show the direction of
the ongoing work: moving from a global QP to local robot-level QPs while keeping
the controller scalable. RDCBF is promising for reducing conservatism, but the
deadlock-resolution part is still a work in progress.

Useful result media:

- `demo/media/formation_control.mp4`
- `demo/media/centralized_cbf_validation.mp4`
- `demo/media/decentralized_cbf_validation.mp4`
- `demo/media/relax_decentralized_cbf_validation.mp4`
- `demo/media/Formation_with_obstacle_and_disturbance.mp4`

## Potential Directions

Several extensions of the current framework are of interest:

1. **Event-triggered communication:** develop an event-triggered communication strategy to improve communication and computational efficiency while preserving the safety guarantees of the CBF framework.

2. **CBFs for uncertain dynamics:** extend the framework to systems with unknown or partially known dynamics by combining CBF-based safety filters with online estimation, adaptive control, or learning-based models.

3. **Scalability for larger multi-robot systems:** study distributed formulations that reduce the computational cost of solving CBF-QPs as the number of agents increases.


## Repository Structure

```text
.
+-- safety_formation/
|   +-- agents/              # Agent and team objects
|   +-- controllers/
|   |   +-- cbf/             # CBF, HOCBF, decentralized CBF, RDCBF logic
|   |   +-- nominal/         # Formation controllers
|   +-- dynamics/            # Single- and double-integrator models
|   +-- formation/           # Topology and graph utilities
|   +-- metrics/             # Safety, formation, and control metrics
|   +-- simulation/          # Scenario runner, disturbances, results
|   +-- visualization/       # Plotting and animation utilities
+-- demo/
|   +-- notebooks/           # Validation and experiment notebooks
|   +-- scripts/             # Runnable script examples
|   +-- media/               # Generated videos
+-- tests/                   # Unit tests and validation tests
+-- requirements.txt
+-- setup.py
+-- README.md
```

## References

[1] W. Ni and D. Cheng, “Leader-following consensus of multi-agent systems under fixed and switching topologies,” *Systems & Control Letters*, vol. 59, no. 3–4, pp. 209–217, 2010, doi: 10.1016/j.sysconle.2010.01.006.

[2] A. D. Ames, S. Coogan, M. Egerstedt, G. Notomista, K. Sreenath, and P. Tabuada, “Control barrier functions: Theory and applications,” in *Proc. 18th Eur. Control Conf. (ECC)*, Naples, Italy, 2019, pp. 3420–3431, doi: 10.23919/ECC.2019.8796030.

[3] L. Wang, A. D. Ames, and M. Egerstedt, “Safety barrier certificates for collision-free multirobot systems,” *IEEE Trans. Robot.*, vol. 33, no. 3, pp. 661–674, 2017, doi: 10.1109/TRO.2017.2659727.

[4] W. Xiao and C. Belta, “High-order control barrier functions,” *IEEE Trans. Autom. Control*, vol. 67, no. 7, pp. 3655–3662, Jul. 2022, doi: 10.1109/TAC.2021.3105491.

Key implementation dependencies include `numpy`, `scipy`, `matplotlib`,
`networkx`, `cvxpy`, `qpsolvers`, `cvxopt`, and `quadprog`.
