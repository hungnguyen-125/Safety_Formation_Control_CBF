# Safe Formation Control for Multi-Robot Systems

This repository implements and validates safety-critical formation control for
multi-robot systems using Control Barrier Functions (CBFs). The project combines
nominal formation controllers with quadratic-programming safety filters so that
robots can track a desired formation while respecting inter-agent safety
distances, actuator limits, and obstacle constraints.

The codebase includes centralized CBF, centralized high-order CBF (HOCBF),
decentralized CBF, and a relaxed decentralized CBF (RDCBF) formulation. The
deadlock-resolution logic is currently under development and should be treated
as an experimental idea rather than a finalized method.

## Overview

Multi-robot formation control has two competing objectives:

- Maintain a desired formation relative to a leader or reference trajectory.
- Avoid unsafe behavior such as robot-robot collision, obstacle collision, and
  excessive control commands.

The main approach in this repository is to separate the controller into two
layers:

1. A nominal formation controller computes the desired acceleration for each
   robot.
2. A CBF safety filter minimally modifies the nominal control by solving a
   quadratic program (QP).

This makes the nominal controller responsible for task performance and the CBF
filter responsible for enforcing safety.

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

## Control Architecture

The implemented architecture is modular:

- `safety_formation/agents/`: agent and team abstractions.
- `safety_formation/formation/`: graph topology, adjacency, Laplacian, and
  augmented Laplacian utilities.
- `safety_formation/controllers/nominal/`: nominal centralized and distributed
  formation controllers.
- `safety_formation/controllers/cbf/`: centralized CBF, HOCBF, decentralized
  CBF, relaxed decentralized CBF logic, QP solvers, and constraint builders.
- `safety_formation/simulation/`: scenario setup, simulation runner,
  disturbances, obstacles, and result containers.
- `safety_formation/metrics/`: safety, formation, and control-effort metrics.
- `safety_formation/visualization/`: plotting and animation helpers.
- `demo/`: notebooks, scripts, and generated media used for validation.

The nominal centralized controller uses the augmented Laplacian `H = L + D_lead`
to combine follower-follower consensus and leader tracking. The CBF layer then
projects the nominal control onto the closest safe control satisfying the active
safety constraints.

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

## How to Run

Create a virtual environment and install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Run the test suite:

```bash
pytest
```

Run the basic simulation script:

```bash
python demo/scripts/basic_simulation.py
```

Open the validation notebooks:

```bash
jupyter notebook demo/notebooks/
```

Some demo scripts and notebooks generate plots or videos, so they may require a
working Matplotlib backend and the required video writer installed on your
system.

## Future Work

- Finish and validate the deadlock-resolution module for RDCBF.
- Add quantitative result tables for minimum distance, formation error, control
  effort, and safety-violation count.
- Improve decentralized simulations with dynamic communication topology.
- Add more obstacle configurations and disturbance models.
- Compare centralized CBF, HOCBF, decentralized CBF, and RDCBF under the same
  benchmark scenarios.
- Reduce debug printing in experimental deadlock code once the method is stable.
- Add documentation for the mathematical derivation and parameter selection.

## References

This project is based on concepts from:

- Multi-agent formation control with graph Laplacian and leader-follower
  consensus.
- Control Barrier Functions for safety-critical control.
- High-Order Control Barrier Functions for systems with higher relative degree.
- Quadratic-programming-based safety filters.
- Decentralized and relaxed CBF formulations for scalable multi-robot systems.

Key implementation dependencies include `numpy`, `scipy`, `matplotlib`,
`networkx`, `cvxpy`, `qpsolvers`, `cvxopt`, and `quadprog`.
