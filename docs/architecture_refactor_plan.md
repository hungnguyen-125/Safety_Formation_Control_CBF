# Architecture Refactor Plan

This document captures the agreed plan for rebuilding the repository structure so the project can scale to modern formation-control methods, multiple robot dynamics, and multiple CBF formulations.

The migration should be staged. Each phase should keep the repository runnable, tested where possible, and committed before moving to the next phase.

## Purpose

The refactor should make it easy to extend the project with:

- multiple dynamics models, such as single integrator, double integrator, unicycle, and future higher-order models
- multiple nominal formation controllers, such as consensus, leader-follower, displacement-based, bearing-based, and other modern formation methods
- multiple CBF safety filters, such as centralized CBF, decentralized CBF, relaxed CBF, high-order CBF, obstacle CBF, and connectivity CBF
- reusable simulation scenarios with obstacles, disturbances, dynamic topology, and sensing limits
- reusable visualization, metrics, and experiment evaluation tools
- lightweight notebooks that configure experiments instead of containing core logic

## Target Package Layout

```text
safety_formation/
  agents/
    agent.py
    team.py

  dynamics/
    base.py
    single_integrator.py
    double_integrator.py
    unicycle.py

  formation/
    topology.py
    graph.py
    references.py

  controllers/
    base.py
    nominal/
      base.py
      consensus.py
      centralized_formation.py
      distributed_formation.py
      pid.py
    cbf/
      base.py
      centralized.py
      decentralized.py
      relaxed.py
      constraints.py
      qp_solver.py
      deadlock_manager.py

  simulation/
    scenario.py
    runner.py
    obstacles.py
    disturbances.py
    results.py
    logger.py

  visualization/
    animation.py
    plotting.py
    topology_plot.py
    trajectory_plot.py

  metrics/
    safety.py
    formation.py
    control.py

  config/
    schemas.py
    presets.py
```

Supporting directories:

```text
demo/
  notebooks/
  media/
  scripts/

tests/
  dynamics/
  agents/
  controllers/
  cbf/
  simulation/
  visualization/
```

## Core Design Rules

The main architectural rule is to separate data, dynamics, control, safety filtering, simulation, and visualization.

```text
Agent:
  Owns id, state, formation target, metadata, and history.
  Does not hard-code a dynamics model.

Dynamics:
  Owns state derivative and integration.
  Defines state_dim and control_dim.

Nominal controller:
  Computes desired control for formation behavior.

CBF:
  Filters nominal control through safety constraints.
  Does not own the simulation loop.

Simulation:
  Owns the time loop, updates, disturbances, logging, and scenario execution.

Visualization:
  Consumes simulation results.
  Does not live inside mathematical/domain objects.
```

## Phase 0: Baseline

Current baseline:

- branch: `refactor/architecture`
- checkpoint commit: `4ff319a` - `Checkpoint before architecture refactor`

Baseline test result recorded before architecture changes:

- `env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest`: collected 0 pytest-style tests
- `.venv/bin/python tests/test_agent.py`: passed
- `.venv/bin/python tests/test_topology.py`: failed on the existing assertion `topo.sees_leader(1) is True`, because the current implementation returns a NumPy boolean instead of a Python `bool`

Before changing structure, run the existing tests:

```bash
pytest
```

If tests fail before the refactor, record those failures as baseline behavior so old issues are not confused with refactor regressions.

## Phase 1: Add Architecture Skeleton

Create the new package layout without moving major logic yet.

Expected changes:

- add `agents/`
- add `dynamics/`
- add `controllers/`
- add `simulation/`
- add `visualization/`
- add `metrics/`
- add `config/`
- add `__init__.py` files
- keep old imports working for now

Goal:

The repository structure should show the intended architecture while the current code still runs.

Suggested commit:

```text
chore: add architecture package skeleton
```

## Phase 2: Extract Dynamics

Current issue:

```text
Agent currently hard-codes double-integrator matrices A and B.
```

Target:

```text
Agent stores state and delegates physics to a Dynamics object.
```

Add:

```text
safety_formation/dynamics/base.py
safety_formation/dynamics/double_integrator.py
safety_formation/dynamics/single_integrator.py
```

Initial interface:

```python
class BaseDynamics:
    state_dim: int
    control_dim: int

    def derivative(self, state, control):
        raise NotImplementedError

    def step(self, state, control, dt):
        return state + self.derivative(state, control) * dt
```

Keep `DoubleIntegrator2D` as the default so existing demos and tests continue to work.

Suggested commit:

```text
refactor: extract agent dynamics interface
```

## Phase 3: Introduce Agent and Team Layer

Move toward:

```text
safety_formation/agents/agent.py
safety_formation/agents/team.py
```

`Agent` should manage:

- id
- state
- formation offset or reference
- dynamics object
- history

`Team` should manage:

- a list of agents
- positions and velocities
- batch state access
- batch updates
- history reset

Goal:

Reduce notebook and simulation-loop complexity by centralizing multi-agent state management.

Suggested commit:

```text
refactor: introduce agent team abstraction
```

## Phase 4: Reorganize Controllers

Current package:

```text
safety_formation/control_law/
```

Target package:

```text
safety_formation/controllers/
```

Target split:

```text
controllers/
  base.py
  nominal/
  cbf/
```

Important interface distinction:

```text
Nominal controllers compute desired formation behavior.
CBF controllers or filters modify nominal control to enforce safety.
```

Temporary compatibility imports should remain in `control_law/` until notebooks and tests are migrated.

Suggested commit:

```text
refactor: reorganize controller modules
```

## Phase 5: Separate CBF Constraints and QP Solving

Current issue:

CBF implementations mix nominal control, constraint generation, QP setup, QP solve, and fallback behavior.

Target split:

```text
controllers/cbf/
  base.py
  constraints.py
  qp_solver.py
  centralized.py
  decentralized.py
  relaxed.py
```

The reusable CBF flow should be:

```text
1. get nominal control
2. build safety constraints
3. solve QP
4. return safe control and optional diagnostic info
```

This phase should prepare the code for:

- pairwise collision CBF
- obstacle CBF
- connectivity CBF
- actuator bounds
- high-order CBF
- relaxed or slack-variable CBF

Suggested commit:

```text
refactor: separate CBF constraints and QP solving
```

## Phase 6: Add Simulation Layer

Move simulation time loops out of notebooks.

Add:

```text
safety_formation/simulation/scenario.py
safety_formation/simulation/runner.py
safety_formation/simulation/obstacles.py
safety_formation/simulation/disturbances.py
safety_formation/simulation/results.py
```

Target notebook usage:

```python
scenario = Scenario(...)
runner = SimulationRunner(scenario)
result = runner.run()
```

Goal:

Experiments should be reproducible from code and not depend on long notebook cells containing core logic.

Suggested commit:

```text
feat: add simulation scenario runner
```

## Phase 7: Clean Up Visualization

Current issue:

Some visualization lives inside domain objects, for example topology plotting inside `Topology`.

Target:

```text
safety_formation/visualization/topology_plot.py
safety_formation/visualization/trajectory_plot.py
safety_formation/visualization/animation.py
```

Rule:

Domain objects should represent mathematical or simulation state. Visualization modules should render those objects or simulation results.

Suggested commit:

```text
refactor: move visualization utilities into package
```

## Phase 8: Add Metrics and Evaluation Tools

Add reusable metrics for comparing formation methods and CBF variants.

Target:

```text
safety_formation/metrics/safety.py
safety_formation/metrics/formation.py
safety_formation/metrics/control.py
```

Useful metrics:

- minimum pairwise distance
- safety violation count
- obstacle clearance
- formation error
- convergence time
- control effort
- deadlock duration
- QP infeasibility count

Suggested commit:

```text
feat: add simulation metrics
```

## Phase 9: Migrate Demos and Notebooks

Target:

```text
demo/
  notebooks/
  media/
  scripts/
```

Notebook rule:

Notebooks should configure, run, visualize, and discuss experiments. They should not contain reusable core logic.

Expected notebook style:

```python
from safety_formation.simulation import Scenario, SimulationRunner
from safety_formation.visualization import animate_trajectories
```

Suggested commit:

```text
refactor: migrate demos to new architecture
```

## Phase 10: Tests and Compatibility Cleanup

Add tests after each major phase.

Priority test areas:

```text
tests/dynamics/
tests/agents/
tests/controllers/
tests/cbf/
tests/simulation/
```

Compatibility policy:

- keep old imports working during migration
- add deprecation comments where useful
- remove old compatibility modules only after notebooks and tests use the new paths

Example old import to preserve temporarily:

```python
from safety_formation.control_law.cbf.decentralized_cbf import DecentralizedCBF
```

Eventually migrate to:

```python
from safety_formation.controllers.cbf import DecentralizedCBF
```

Suggested commit:

```text
test: add coverage for refactored architecture
```

## Refactor Execution Checklist

For each phase:

```bash
pytest
git status
git add -A
git commit -m "<phase commit message>"
```

Use small commits. If a phase becomes too large, split it.

## Recommended First Step

Start with only:

1. Phase 1: add architecture skeleton
2. Phase 2: extract dynamics while preserving current behavior

Do not migrate notebooks or rewrite CBF internals until the dynamics and package boundaries are stable.
