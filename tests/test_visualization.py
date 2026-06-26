import matplotlib

matplotlib.use("Agg")

import numpy as np

from safety_formation import Agent, Topology
from safety_formation.controllers.cbf import CBFTopology
from safety_formation.visualization import plot_cbf_topology, plot_topology


def test_static_topology_plot_function_and_compatibility_method():
    topology = Topology(
        num_agents=2,
        adjacency_matrix=np.array([[0.0, 1.0], [1.0, 0.0]]),
        leader_access=[1.0, 0.0],
    )

    ax_new = plot_topology(topology)
    ax_old = topology.plot()

    assert ax_new.get_title() == "Topology (Leader=0, Followers=1..N)"
    assert ax_old.get_title() == "Topology (Leader=0, Followers=1..N)"


def test_cbf_topology_plot_function_and_compatibility_method():
    agents = [
        Agent(agent_id=1, x0=[0.0, 0.0, 0.0, 0.0], f_target=[0.0, 0.0, 0.0, 0.0]),
        Agent(agent_id=2, x0=[1.0, 0.0, 0.0, 0.0], f_target=[0.0, 0.0, 0.0, 0.0]),
    ]
    topology = CBFTopology(agent_list=agents, d_min=0.5, gamma=0.1)

    ax_new = plot_cbf_topology(topology, show_radius=True)
    ax_old = topology.plot(show_radius=True)

    assert ax_new.get_title() == "CBF Dynamic Neighborhood Topology"
    assert ax_old.get_title() == "CBF Dynamic Neighborhood Topology"
