import numpy as np

from safety_formation import Agent
from safety_formation.simulation import (
    ConstantControlDisturbance,
    Scenario,
    SimulationRunner,
)


def make_agents():
    return [
        Agent(agent_id=1, x0=[0.0, 0.0, 0.0, 0.0], f_target=[0.0, 0.0, 0.0, 0.0]),
        Agent(agent_id=2, x0=[1.0, 0.0, 0.0, 0.0], f_target=[0.0, 0.0, 0.0, 0.0]),
    ]


def constant_policy(step, time, agents, topology):
    return np.array([
        [1.0, 0.0],
        [0.0, -1.0],
    ])


def test_simulation_runner_executes_control_policy():
    scenario = Scenario(
        agents=make_agents(),
        dt=0.1,
        steps=2,
        control_policy=constant_policy,
    )

    result = SimulationRunner(scenario).run()

    assert result.times.shape == (3,)
    assert result.states.shape == (3, 2, 4)
    assert result.controls.shape == (2, 2, 2)
    assert np.allclose(result.final_state, [
        [0.01, 0.0, 0.2, 0.0],
        [1.0, -0.01, 0.0, -0.2],
    ])
    assert np.allclose(result.positions[-1], [[0.01, 0.0], [1.0, -0.01]])


def test_simulation_runner_applies_control_disturbance():
    scenario = Scenario(
        agents=make_agents(),
        dt=0.1,
        steps=1,
        control_policy=constant_policy,
        disturbances=[ConstantControlDisturbance([[1.0, 0.0], [0.0, 1.0]])],
    )

    result = SimulationRunner(scenario).run()

    assert np.allclose(result.controls[0], [[2.0, 0.0], [0.0, 0.0]])
    assert np.allclose(result.final_state, [
        [0.0, 0.0, 0.2, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ])
