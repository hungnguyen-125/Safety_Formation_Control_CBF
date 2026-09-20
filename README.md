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
    style="width: 80%; max-width: 800px; height: auto;"
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

The safety filter solves a QP that keeps the applied control input as close as possible to the nominal controller while enforcing safety constraints.

## Framework

The implemented framework considers:

- inter-agent collision avoidance;
- obstacle avoidance;
- actuator and velocity limits;
- centralized and distributed safety-filter architectures.

A common simulation environment is used to compare the different CBF formulations in terms of behavior, conservativeness, and computational characteristics under the same multi-robot scenarios.

## Problem Formulation

Each robot is modeled as a planar double-integrator system:

$$
\dot{p}_i = v_i
$$

$$
\dot{v}_i = u_i
$$

where $p_i$, $v_i$, and $u_i$ denote the robot position, velocity, and acceleration input.

The nominal controller drives each robot toward its assigned target or formation reference. For formation-control scenarios, the tracking error is defined as:

$$
e_i = x_i - f_i - x_L
$$

where

$$
x_i =
\begin{bmatrix}
p_x & p_y & v_x & v_y
\end{bmatrix}^{T},
$$

$f_i$ is the desired formation offset, and $x_L$ is the leader or reference state.

The safety objective is to maintain a prescribed minimum distance between neighboring robots:

$$
\|p_i - p_j\| \ge d_{\min}
$$

For circular static obstacles, the safety condition is:

$$
\|p_i - p_{\mathrm{obs}}\| \ge r_{\mathrm{obs}} + d_{\min}
$$

The nominal controller is combined with a safety filter that modifies the control input only when required to satisfy collision-avoidance and actuator constraints.

---

## Safety Controllers / Filters

#### Conventional Centralized CBF

The conventional centralized CBF is based on a pairwise braking-distance interpretation of collision avoidance. For each neighboring pair of robots, safety is evaluated by checking whether their relative closing velocity can be reduced to zero before the inter-agent distance reaches the prescribed safety distance.

The underlying safety condition is written as:

$$
\|p_{ij}\| + \int_{t_0}^{t_0 + T_b} \bar{v}_{ij}(t)\,dt \ge d_{\min},
\qquad \forall i \neq j
$$

where

$$
p_{ij} = p_i - p_j,
\qquad
v_{ij} = v_i - v_j
$$

and $\bar{v}_{ij}$ denotes the component of the relative velocity $v_{ij}$ along the inter-agent direction. $T_b$ is the braking time required to reduce the closing velocity to zero, and $d_{\min}$ is the prescribed minimum safety distance.

This formulation leads to a velocity-aware barrier function that accounts for both the current inter-agent distance and how quickly the robots are approaching each other. Since relative velocity is already included in the barrier definition, differentiating the barrier introduces the acceleration control input directly into the resulting CBF constraint.

A detailed derivation of this formulation is provided in the [M1 Safe Formation Control Report](./docs/report/Safe_Formation_Control_M1_Report.pdf).

#### Centralized HOCBF

The HOCBF formulation starts from a purely position-based barrier:

$$
h_{ij} = \|p_{ij}\|^2 - d_{\min}^2.
$$

For the double-integrator dynamics, this barrier has relative degree two with respect to the acceleration input. After differentiating the barrier twice and applying the recursive HOCBF condition with gains $k_1$ and $k_2$, the final safety constraint is obtained as:

$$
2p_{ij}^{T}(u_i-u_j)
+
2\|v_{ij}\|^2
+
(k_1+k_2)\dot{h}_{ij}
+
k_1k_2h_{ij}
\ge 0,
$$

where

$$
p_{ij}=p_i-p_j,
\qquad
v_{ij}=v_i-v_j.
$$

This constraint can then be directly incorporated into the centralized QP.

---

### Relaxed Decentralized CBF (RDCBF)

The RDCBF formulation replaces the centralized optimization problem with one local QP per robot:

$$
\begin{aligned}
\min_{u_i} \quad & \|u_i-u_{\mathrm{nom},i}\|^2 \\
\text{s.t.} \quad & \text{local CBF constraints}, \\
& \text{actuator limits}.
\end{aligned}
$$

Each robot computes its safe control using only information from its local neighbors. This decentralized structure reduces the computational burden of solving a single large centralized optimization problem, which becomes increasingly expensive as the number of agents grows.

However, unlike the centralized formulation, each robot does not have access to the complete control vector of the multi-agent system. It only observes the states of its neighboring agents and must therefore enforce safety without directly knowing their future control actions. This makes the allocation of collision-avoidance responsibility between neighboring robots more challenging and can lead to more conservative local CBF constraints. The relaxed formulation introduces additional variables that modify the effective CBF gains associated with neighboring constraints. This provides extra flexibility in configurations where a strict decentralized CBF would otherwise become overly restrictive or lead to poor motion behavior.

A more detailed derivation of the RDCBF formulation is provided in the [Safe Formation Control M1 Report](./docs/report/Safe_Formation_Control_M1_Report.pdf).

The deadlock-resolution component is not fully developed and still under development.

## Simulation Scenarios

Two representative scenarios are used to evaluate the safety controllers.

### 1. Position Swap

Multiple agents are initially placed uniformly on a circle, with each agent assigned a target on the opposite side. As all agents move toward their targets simultaneously, their nominal trajectories intersect near the center, creating a highly congested region with a high risk of collisions.

This scenario is used to evaluate inter-agent collision avoidance, minimum-distance preservation, and the intervention behavior of the CBF and HOCBF safety filters.

### 2. Formation Control with Static Obstacles

Multiple agents are required to move toward a desired formation while avoiding static obstacles in the workspace. The nominal controller drives the agents toward their assigned formation positions, while the safety filter modifies the control inputs whenever necessary to prevent collisions.

This scenario evaluates both inter-agent and agent-obstacle safety constraints while preserving the formation objective.

---
## Selected Results

### 1. Centralized CBF and HOCBF

#### Position Swapping

Despite being derived from different formulations, the two controllers exhibit very similar behavior in this experiment. The reason might be although these two approaches are formulated differently, both ultimately impose safety constraints that depend on relative position, relative velocity, and acceleration. This leads to very similar avoidance behavior when both controllers are tuned close to the safety boundary.

<p align="center">
  <table>
    <tr>
      <td align="center">
        <img src="docs/figures/centralized_cbf_validation_dynamics_topology.gif" width="100%">
        <br>
        <sub><b>Centralized CBF</b></sub>
      </td>
      <td align="center">
        <img src="docs/figures/centralized_hocbf_validation.gif" width="100%">
        <br>
        <sub><b>Centralized HOCBF</b></sub>
      </td>
    </tr>
  </table>
</p>


### 3. Relaxed Distributed CBF (RDCBF)

#### Position Swapping

The RDCBF controller enables distributed collision avoidance during the position-swapping task while preserving progress toward the agents' individual targets.

<p align="center">
  <img src="docs/figures/relax_distributed_cbf_validation.gif" width="600">
</p>


#### Formation Control with Static Obstacles

The RDCBF controller is further evaluated in a formation-control scenario with static obstacles. The agents maintain collision avoidance with both neighboring agents and obstacles while converging toward the desired formation.

<p align="center">
  <img src="docs/figures/formation_with_obstacle_and_disturbance.gif" width="800">
</p>

### Useful Result Media

| Video | Description |
|---|---|
| [Centralized CBF Formation Control](demo/media/centralized_cbf_formation_control.mp4) | Centralized CBF applied to formation control. |
| [Centralized CBF with Dynamic Topology](demo/media/centralized_cbf_validation_dynamics_topology.mp4) | Validation of the centralized CBF with dynamic topology. |
| [Centralized HOCBF](demo/media/centralized_hocbf_validation.mp4) | Validation of the centralized HOCBF controller. |
| [Distributed CBF](demo/media/distributed_cbf_validation.mp4) | Validation of the distributed CBF formulation. |
| [Formation with Obstacles and Disturbance](demo/media/formation_with_obstacle_and_disturbance.mp4) | Formation control with static obstacles and disturbances. |
| [Nominal Formation Controller](demo/media/nominal_formation_controller.mp4) | Baseline formation control without a safety filter. |
| [Relaxed Distributed CBF](demo/media/relax_distributed_cbf_validation.mp4) | Validation of the relaxed distributed CBF formulation. |
---
## Potential Directions

Several extensions of the current framework are of interest:

1. **Event-triggered communication:** develop an event-triggered communication strategy to improve communication and computational efficiency while preserving the safety guarantees of the CBF framework.

2. **CBFs for uncertain dynamics:** extend the framework to systems with unknown or partially known dynamics by combining CBF-based safety filters with online estimation, adaptive control, or learning-based models.

3. **Scalability for larger multi-robot systems:** study distributed formulations that reduce the computational cost of solving CBF-QPs as the number of agents increases.

---
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
---
## References

[1] W. Ni and D. Cheng, “Leader-following consensus of multi-agent systems under fixed and switching topologies,” *Systems & Control Letters*, vol. 59, no. 3–4, pp. 209–217, 2010, doi: 10.1016/j.sysconle.2010.01.006.

[2] A. D. Ames, S. Coogan, M. Egerstedt, G. Notomista, K. Sreenath, and P. Tabuada, “Control barrier functions: Theory and applications,” in *Proc. 18th Eur. Control Conf. (ECC)*, Naples, Italy, 2019, pp. 3420–3431, doi: 10.23919/ECC.2019.8796030.

[3] L. Wang, A. D. Ames, and M. Egerstedt, “Safety barrier certificates for collision-free multirobot systems,” *IEEE Trans. Robot.*, vol. 33, no. 3, pp. 661–674, 2017, doi: 10.1109/TRO.2017.2659727.

[4] W. Xiao and C. Belta, “High-order control barrier functions,” *IEEE Trans. Autom. Control*, vol. 67, no. 7, pp. 3655–3662, Jul. 2022, doi: 10.1109/TAC.2021.3105491.

