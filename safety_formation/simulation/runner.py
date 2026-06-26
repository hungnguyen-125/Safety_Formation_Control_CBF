import numpy as np

from safety_formation.agents import Team

from .results import SimulationResult


class SimulationRunner:
    def __init__(self, scenario):
        self.scenario = scenario
        self.team = Team(scenario.agents)

    def run(self):
        scenario = self.scenario
        times = [0.0]
        states = [self._snapshot_states()]
        controls = []

        for step in range(scenario.steps):
            time = step * scenario.dt
            control = self._compute_control(step, time)
            control = self._apply_disturbances(control, step, time)

            self.team.update_physics(control, scenario.dt)

            controls.append(np.asarray(control, dtype=float).copy())
            times.append((step + 1) * scenario.dt)
            states.append(self._snapshot_states())

        return SimulationResult(
            times=np.asarray(times, dtype=float),
            states=np.asarray(states, dtype=float),
            controls=np.asarray(controls, dtype=float),
        )

    def _compute_control(self, step, time):
        scenario = self.scenario

        if scenario.control_policy is not None:
            return scenario.control_policy(
                step=step,
                time=time,
                agents=scenario.agents,
                topology=scenario.topology,
            )

        if scenario.nominal_controller is None:
            raise ValueError("Scenario requires control_policy or nominal_controller.")

        nominal = scenario.nominal_controller.compute_nominal(
            scenario.agents,
            scenario.topology,
            leader_state=scenario.leader_state,
        )

        if scenario.safety_filter is None:
            return nominal

        return scenario.safety_filter.compute_safe_control(
            scenario.agents,
            scenario.topology,
            nominal,
            obstacles=scenario.obstacles,
        )

    def _apply_disturbances(self, control, step, time):
        disturbed = np.asarray(control, dtype=float)

        for disturbance in self.scenario.disturbances:
            disturbed = disturbance.apply(
                control=disturbed,
                step=step,
                time=time,
                agents=self.scenario.agents,
            )

        return disturbed

    def _snapshot_states(self):
        return np.asarray([agent.state.flatten() for agent in self.scenario.agents])
