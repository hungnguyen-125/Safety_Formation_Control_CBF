import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def plot_topology(topology, leader_pos="center", radius=3.0, ax=None, seed=2):
    """
    Plot static formation topology with follower indices 1..N and optional leader node 0.
    """
    has_leader = np.any(topology.D_lead)

    graph = nx.from_numpy_array(topology.adj_matrix)
    graph = nx.relabel_nodes(graph, {i: i + 1 for i in range(topology.n)})

    if has_leader:
        graph.add_node(0)
        for i in range(topology.n):
            if topology.D_lead[i, i] > 0:
                graph.add_edge(0, i + 1)

    positions = {}
    angles = np.linspace(0, 2 * np.pi, topology.n, endpoint=False)
    for i, angle in enumerate(angles, start=1):
        positions[i] = (radius * np.cos(angle), radius * np.sin(angle))

    if has_leader:
        if leader_pos == "center":
            positions[0] = (0.0, 0.0)
        elif leader_pos == "bottom":
            positions[0] = (0.0, -1.4 * radius)
        else:
            raise ValueError("leader_pos must be 'center' or 'bottom'")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    node_colors = [
        "red" if (has_leader and node == 0) else "skyblue"
        for node in graph.nodes()
    ]

    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color=node_colors,
        node_size=900,
        edgecolors="k",
        linewidths=1,
        ax=ax,
    )
    nx.draw_networkx_labels(graph, positions, font_size=12, ax=ax)

    follower_edges = [
        (u, v) for (u, v) in graph.edges() if (u != 0 and v != 0)
    ]
    nx.draw_networkx_edges(graph, positions, edgelist=follower_edges, width=2, ax=ax)

    if has_leader:
        leader_edges = [
            (u, v) for (u, v) in graph.edges() if (u == 0 or v == 0)
        ]
        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=leader_edges,
            width=2,
            style="dashed",
            ax=ax,
        )
        ax.set_title("Topology (Leader=0, Followers=1..N)")
    else:
        ax.set_title("Topology (Followers only)")

    ax.axis("off")
    return ax


def plot_cbf_topology(cbf_topology, ax=None, show_radius=False):
    """
    Plot dynamic CBF neighborhood topology using current agent positions.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    graph = nx.Graph()
    positions = {}

    for i, agent in enumerate(cbf_topology.agent_list):
        node_id = i + 1
        graph.add_node(node_id)
        pos = agent.pos.flatten()
        positions[node_id] = (pos[0], pos[1])

    for i in range(cbf_topology.n):
        for j in range(i + 1, cbf_topology.n):
            if cbf_topology.adj_matrix[i, j] > 0 or cbf_topology.adj_matrix[j, i] > 0:
                graph.add_edge(i + 1, j + 1)

    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color="skyblue",
        node_size=700,
        edgecolors="k",
        ax=ax,
    )
    nx.draw_networkx_labels(graph, positions, font_size=11, ax=ax)
    nx.draw_networkx_edges(graph, positions, width=2, ax=ax)

    if show_radius:
        for i, agent in enumerate(cbf_topology.agent_list):
            pos = agent.pos.flatten()
            radius = cbf_topology.neighborhood_radii[i]
            circle = plt.Circle(
                (pos[0], pos[1]),
                radius,
                fill=False,
                linestyle="--",
                alpha=0.35,
            )
            ax.add_patch(circle)

    ax.set_title("CBF Dynamic Neighborhood Topology")
    ax.set_aspect("equal")
    ax.grid(True)
    return ax
