import numpy as np

from safety_formation import Agent, Team
from safety_formation.agents import Agent as CanonicalAgent
from safety_formation.formation.agent import Agent as CompatibleAgent


def make_agents():
    return [
        Agent(agent_id=1, x0=[0.0, 0.0, 0.0, 0.0], f_target=[1.0, 0.0, 0.0, 0.0]),
        Agent(agent_id=2, x0=[1.0, 0.0, 0.0, 0.0], f_target=[0.0, 1.0, 0.0, 0.0]),
    ]


def test_agent_import_paths_are_compatible():
    assert Agent is CanonicalAgent
    assert CompatibleAgent is CanonicalAgent


def test_team_stacks_states_positions_and_velocities():
    team = Team(make_agents())

    assert len(team) == 2
    assert team.states.shape == (8, 1)
    assert team.formation_offsets.shape == (8, 1)
    assert np.allclose(team.positions, [[0.0, 0.0], [1.0, 0.0]])
    assert np.allclose(team.velocities, [[0.0, 0.0], [0.0, 0.0]])


def test_team_updates_all_agents_and_histories():
    team = Team(make_agents())

    team.update_physics(controls=np.array([[1.0, 0.0], [0.0, -1.0]]), dt=0.1)

    assert np.allclose(team[0].state.flatten(), [0.0, 0.0, 0.1, 0.0])
    assert np.allclose(team[1].state.flatten(), [1.0, 0.0, 0.0, -0.1])
    assert all(history.shape == (1, 4) for history in team.get_histories())

    team.clear_history()

    assert all(history.size == 0 for history in team.get_histories())
