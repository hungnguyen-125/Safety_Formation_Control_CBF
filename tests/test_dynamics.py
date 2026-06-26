import numpy as np

from safety_formation.dynamics import DoubleIntegrator2D, SingleIntegrator2D
from safety_formation.formation.agent import Agent


def test_double_integrator_step_matches_existing_agent_physics():
    dynamics = DoubleIntegrator2D()
    state = np.array([0.0, 0.0, 0.0, 0.0]).reshape(4, 1)
    control = np.array([1.0, -0.5]).reshape(2, 1)

    next_state = dynamics.step(state, control, dt=0.1)

    assert np.allclose(next_state.flatten(), [0.0, 0.0, 0.1, -0.05])


def test_agent_uses_double_integrator_by_default():
    agent = Agent(
        agent_id=1,
        x0=[0.0, 0.0, 0.0, 0.0],
        f_target=[1.0, 0.5, 0.0, 0.0],
    )

    agent.update_physics(u=[1.0, -0.5], dt=0.1)

    assert isinstance(agent.dynamics, DoubleIntegrator2D)
    assert np.allclose(agent.state.flatten(), [0.0, 0.0, 0.1, -0.05])
    assert np.allclose(agent.pos.flatten(), [0.0, 0.0])
    assert np.allclose(agent.vel.flatten(), [0.1, -0.05])


def test_agent_accepts_single_integrator_dynamics():
    agent = Agent(
        agent_id=1,
        x0=[0.0, 0.0],
        f_target=[1.0, 0.5],
        dynamics=SingleIntegrator2D(),
    )

    agent.update_physics(u=[1.0, -0.5], dt=0.1)

    assert np.allclose(agent.state.flatten(), [0.1, -0.05])
    assert np.allclose(agent.pos.flatten(), [0.1, -0.05])
    assert np.allclose(agent.vel.flatten(), [0.0, 0.0])
