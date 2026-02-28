import numpy as np
import matplotlib.pyplot as plt

def plot_formation_error(norms, dt, title=r"Formation error $\|\delta_i\|$", caption=None):
    """
    norms: (T,N) matrix
    """
    T, N = norms.shape
    t = np.arange(T) * dt

    plt.figure(figsize=(8.6, 4.6))  # gần giống paper
    for i in range(N):
        plt.plot(t, norms[:, i], linewidth=2, label=f"agent {i+1}")

    plt.xlabel("Time (s)")
    plt.ylabel(r"$\|\delta_i\|$")
    plt.xlim(t[0], t[-1])
    plt.grid(True, alpha=0.3)

    plt.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="black")
    plt.title(title)

    if caption is not None:
        # caption kiểu paper (tuỳ bạn có muốn)
        plt.figtext(0.5, -0.05, caption, ha="center", fontsize=12)

    plt.tight_layout()
    plt.show()
    
def plot_relative_trajectories(agents, leader, topology, show_topology_at="final"):
    """
    Plot relative trajectories (p_i - p_0) in 2D, with dashed topology connections.

    Parameters
    ----------
    agents : list[Agent]
        Followers (IDs 1..N), each has history with [x,y,vx,vy].
    leader : Agent
        Leader (id=0), has history.
    topology : Topology
        Has adj_matrix (NxN) and get_neighbors(agent_id) returning IDs 1..N.
    show_topology_at : {"final", "initial", int}
        - "final": draw dashed edges using final positions
        - "initial": using initial positions
        - int: specific time index k
    """
    # --- Collect histories ---
    N = len(agents)
    T = len(agents[0].history)
    leader_hist = np.array(leader.history)[:, :2]  # (T,2)

    rel_traj = []
    for ag in agents:
        x = np.array(ag.history)[:, :2]           # (T,2)
        rel_traj.append(x - leader_hist)          # (T,2)
    rel_traj = np.stack(rel_traj, axis=1)         # (T,N,2)

    # Choose time index for topology overlay
    if show_topology_at == "final":
        k = T - 1
    elif show_topology_at == "initial":
        k = 0
    elif isinstance(show_topology_at, int):
        k = int(show_topology_at)
        k = max(0, min(T - 1, k))
    else:
        raise ValueError("show_topology_at must be 'final', 'initial', or an int index")

    # Positions at chosen time for dashed edges
    Pk = rel_traj[k]  # (N,2)

    # --- Plot ---
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    markers = ['o', '^', '>', '+', 's', '*', 'D', 'v', '<', 'x']  # cycle if N>len
    for i, ag in enumerate(agents):
        m = markers[i % len(markers)]
        ax.plot(rel_traj[:, i, 0], rel_traj[:, i, 1], linewidth=2, marker=m,
                markevery=[0, k], markersize=8, label=f"agent {ag.id}")

    # Leader at origin (relative frame)
    ax.plot(0, 0, marker='*', markersize=14, label="leader")

    # --- Dashed topology edges (between followers) ---
    # adj_matrix is 0-based (idx 0..N-1). follower id = idx+1
    A = topology.adj_matrix
    for i in range(N):
        for j in range(i+1, N):
            if A[i, j] > 0 or A[j, i] > 0:
                ax.plot([Pk[i, 0], Pk[j, 0]], [Pk[i, 1], Pk[j, 1]],
                        linestyle='--', linewidth=1, alpha=0.6)

    ax.set_xlabel(r"$p_x$")
    ax.set_ylabel(r"$p_y$")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    ax.legend(loc="center right", frameon=True, edgecolor="black")
    ax.set_title("Relative trajectories to the leader (dashed: topology links)")
    plt.tight_layout()
    plt.show()
    
def plot_world_trajectories(agents, leader, topology=None, topology_at="final"):
    N = len(agents)
    T = len(agents[0].history)

    leader_xy = np.array(leader.history)[:, :2]  # (T,2)

    # time index for topology overlay (optional)
    if topology_at == "final":
        k = T-1
    elif topology_at == "initial":
        k = 0
    else:
        k = int(topology_at)

    plt.figure(figsize=(10,8))
    ax = plt.gca()

    # followers
    markers = ['o','^','>','+','s','*','D','v','<','x']
    for i, ag in enumerate(agents):
        xy = np.array(ag.history)[:, :2]
        ax.plot(xy[:,0], xy[:,1], linewidth=2, marker=markers[i % len(markers)],
                markevery=[0, k], markersize=8, label=f"agent {ag.id}")

    # leader (world)
    ax.plot(leader_xy[:,0], leader_xy[:,1], linewidth=2, marker='*',markersize=14, markevery=[0, k], label="leader")

    # optional: dashed topology links at time k
    if topology is not None:
        Pk = np.array([np.array(ag.history)[k,:2] for ag in agents])  # (N,2)
        A = topology.adj_matrix
        for i in range(N):
            for j in range(i+1, N):
                if A[i,j] > 0 or A[j,i] > 0:
                    ax.plot([Pk[i,0],Pk[j,0]], [Pk[i,1],Pk[j,1]], '--', linewidth=1, alpha=0.6)

    ax.set_xlabel(r"$p_x$")
    ax.set_ylabel(r"$p_y$")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", frameon=True, edgecolor="black")
    ax.set_title("World trajectories (leader moves)")
    plt.tight_layout()
    plt.show()