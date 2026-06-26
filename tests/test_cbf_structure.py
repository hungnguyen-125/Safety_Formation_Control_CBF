import numpy as np

from safety_formation import Agent
from safety_formation.controllers.cbf import (
    build_centralized_constraints,
    build_decentralized_constraints,
)


class FullTopology:
    adj_matrix = np.array([[0.0, 1.0], [1.0, 0.0]])


def make_agents():
    return [
        Agent(agent_id=1, x0=[0.0, 0.0, 0.0, 0.0], f_target=[0.0, 0.0, 0.0, 0.0]),
        Agent(agent_id=2, x0=[2.0, 0.0, 0.0, 0.0], f_target=[0.0, 0.0, 0.0, 0.0]),
    ]


def test_centralized_constraint_builder_returns_qp_matrices():
    agents = make_agents()

    G, h = build_centralized_constraints(
        all_agents=agents,
        topology=FullTopology(),
        d_min=0.5,
        gamma=0.1,
    )

    assert G.shape == (9, 4)
    assert h.shape == (9,)
    assert np.allclose(G[-8:], np.array([
        [1.0, 0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, -1.0],
    ]))


def test_decentralized_constraint_builder_returns_local_qp_matrices():
    agents = make_agents()

    G, h = build_decentralized_constraints(
        agent_i=agents[0],
        agent_id=1,
        all_agents=agents,
        neighbor_list=[2],
        d_min=0.5,
        gamma=0.1,
        p=3,
    )

    assert G.shape == (6, 3)
    assert h.shape == (6,)
    assert np.allclose(G[-5:], np.array([
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
    ]))
