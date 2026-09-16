import numpy as np
import matplotlib.pyplot as plt

import matplotlib.animation as animation

from matplotlib.patches import Circle

def plot_formation_error(norms, dt, title=r"Formation error $\|\delta_i\|$", caption=None):
    """
    norms: (T,N) matrix
    """
    T, N = norms.shape
    t = np.arange(T) * dt

    plt.figure(figsize=(8.6, 4.6))  
    for i in range(N):
        plt.plot(t, norms[:, i], linewidth=2, label=f"agent {i+1}")

    plt.xlabel("Time (s)")
    plt.ylabel(r"$\|\delta_i\|$")
    plt.xlim(t[0], t[-1])
    plt.grid(True, alpha=0.3)

    plt.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="black")
    plt.title(title)

    if caption is not None:
        # Optional paper-style caption
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
    
from matplotlib.patches import Circle
import numpy as np
import matplotlib.pyplot as plt

def plot_world_trajectories(
        agents,
        leader,
        topology=None,
        topology_at="final",
        obstacles=None,
        d_min=None):

    N = len(agents)
    T = len(agents[0].history)

    leader_xy = np.array(leader.history)[:, :2]

    if topology_at == "final":
        k = T - 1
    elif topology_at == "initial":
        k = 0
    else:
        k = int(topology_at)

    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    # Followers
    markers = ['o', '^', '>', '+', 's', '*', 'D', 'v', '<', 'x']

    for i, ag in enumerate(agents):
        xy = np.array(ag.history)[:, :2]

        ax.plot(
            xy[:, 0],
            xy[:, 1],
            linewidth=2,
            marker=markers[i % len(markers)],
            markevery=[0, k],
            markersize=8,
            label=f"agent {ag.id}"
        )

    # Leader
    ax.plot(
        leader_xy[:, 0],
        leader_xy[:, 1],
        linewidth=2,
        marker='*',
        markersize=14,
        markevery=[0, k],
        label="leader"
    )

    # Obstacles
    if obstacles is not None:
        for idx, obs in enumerate(obstacles):
            obs_center = np.asarray(obs["center"], dtype=float).reshape(-1)[:2]
            obs_radius = float(obs["radius"])

            obs_patch = Circle(
                obs_center,
                radius=obs_radius,
                fill=True,
                alpha=0.35,
                color="black",
                label="obstacle" if idx == 0 else None
            )
            ax.add_patch(obs_patch)

            # Optional inflated safety zone
            if d_min is not None:
                obs_safe_patch = Circle(
                    obs_center,
                    radius=obs_radius + d_min,
                    fill=False,
                    linestyle="--",
                    alpha=0.5,
                    color="black",
                    label="obstacle safety zone" if idx == 0 else None
                )
                ax.add_patch(obs_safe_patch)

    # Optional topology links at time k
    if topology is not None:
        Pk = np.array([np.array(ag.history)[k, :2] for ag in agents])
        A = topology.adj_matrix

        for i in range(N):
            for j in range(i + 1, N):
                if A[i, j] > 0 or A[j, i] > 0:
                    ax.plot(
                        [Pk[i, 0], Pk[j, 0]],
                        [Pk[i, 1], Pk[j, 1]],
                        '--',
                        linewidth=1,
                        alpha=0.6
                    )

    ax.set_xlabel(r"$p_x$")
    ax.set_ylabel(r"$p_y$")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", frameon=True, edgecolor="black")
    ax.set_title("World trajectories (leader moves)")

    plt.tight_layout()
    plt.show()


from matplotlib.patches import Circle
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np

def generate_formation_video(
        agents,
        leader,
        dt,
        T,
        d_min,
        filename,
        obstacles=None,
        wind_x_range=None,      # example: (3.0, 7.0)
        arrow_mode="velocity",
        arrow_length=0.35,
        graph_title="World trajectories (leader moves)"):

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.set_xlim(-3, T + 2)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='-', alpha=0.3)
    ax.set_xlabel(r'$p_x$')
    ax.set_ylabel(r'$p_y$')
    ax.set_title(graph_title)

    # Disturbance region fixed in space
    if wind_x_range is not None:
        x_start, x_end = wind_x_range

        ax.axvspan(
            x_start,
            x_end,
            color="gold",
            alpha=0.18,
            zorder=-100,
        )

        ax.text(
            (x_start + x_end) / 2,
            ax.get_ylim()[1] * 0.88,
            "Disturbance region",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            bbox=dict(
                facecolor="gold",
                alpha=0.45,
                edgecolor="none"
            )
        )

    styles = {
        1: {'color': 'tab:blue',   'marker': 'o'},
        2: {'color': 'tab:orange', 'marker': '^'},
        3: {'color': 'tab:green',  'marker': '>'},
        4: {'color': 'tab:red',    'marker': '+'},
        5: {'color': 'tab:purple', 'marker': 's'},
        6: {'color': 'tab:brown',  'marker': '*'}
    }

    follower_plots = []
    follower_trajs = []
    safety_circles = []
    direction_arrows = []
    obstacle_patches = []

    # Static obstacles
    if obstacles is not None:
        for obs in obstacles:
            obs_center = np.asarray(obs["center"], dtype=float).reshape(-1)[:2]
            obs_radius = float(obs["radius"])

            obs_patch = Circle(
                obs_center,
                radius=obs_radius,
                fill=True,
                alpha=0.35,
                color="black",
                label="obstacle" if len(obstacle_patches) == 0 else None
            )
            ax.add_patch(obs_patch)
            obstacle_patches.append(obs_patch)

            obs_safe_patch = Circle(
                obs_center,
                radius=obs_radius + d_min,
                fill=False,
                linestyle="--",
                alpha=0.5,
                color="black"
            )
            ax.add_patch(obs_safe_patch)
            obstacle_patches.append(obs_safe_patch)

    # Followers
    for ag in agents:
        s = styles[ag.id]

        p, = ax.plot(
            [], [],
            color=s['color'],
            marker=s['marker'],
            markersize=8,
            label=f'agent {ag.id}'
        )
        follower_plots.append(p)

        t, = ax.plot(
            [], [],
            color=s['color'],
            linewidth=2,
            alpha=0.8
        )
        follower_trajs.append(t)

        c = Circle(
            (0, 0),
            radius=d_min,
            fill=False,
            linestyle='--',
            alpha=0.5,
            color=s['color']
        )
        ax.add_patch(c)
        safety_circles.append(c)

        arrow = ax.quiver(
            [0], [0],
            [0], [0],
            color=s['color'],
            angles='xy',
            scale_units='xy',
            scale=1,
            width=0.006
        )
        direction_arrows.append(arrow)

    # Leader
    leader_plot, = ax.plot(
        [], [],
        color='tab:pink',
        marker='*',
        markersize=15,
        label='leader'
    )

    leader_traj, = ax.plot(
        [], [],
        color='tab:pink',
        linewidth=2,
        alpha=0.8
    )

    ax.legend(loc='upper left', shadow=True)

    num_frames = len(agents[0].history)

    def update(frame):
        l_hist = np.asarray(leader.history)
        leader_pos = l_hist[frame, :2]

        leader_plot.set_data(
            [leader_pos[0]],
            [leader_pos[1]]
        )

        leader_traj.set_data(
            l_hist[:frame + 1, 0],
            l_hist[:frame + 1, 1]
        )

        for i, ag in enumerate(agents):
            ag_hist = np.asarray(ag.history)

            pos = ag_hist[frame, :2]
            vel = ag_hist[frame, 2:4]

            follower_plots[i].set_data(
                [pos[0]],
                [pos[1]]
            )

            follower_trajs[i].set_data(
                ag_hist[:frame + 1, 0],
                ag_hist[:frame + 1, 1]
            )

            safety_circles[i].center = (pos[0], pos[1])

            if arrow_mode == "velocity":
                direction = vel

            elif arrow_mode == "target":
                f_i = np.asarray(ag.f, dtype=float).reshape(-1)[:2]
                target_pos = leader_pos + f_i
                direction = target_pos - pos

            else:
                direction = np.zeros(2)

            norm_dir = np.linalg.norm(direction)

            if norm_dir > 1e-6:
                direction = direction / norm_dir
            else:
                direction = np.zeros(2)

            direction_arrows[i].set_offsets([pos])
            direction_arrows[i].set_UVC(
                arrow_length * direction[0],
                arrow_length * direction[1]
            )

        return (
            follower_plots
            + follower_trajs
            + safety_circles
            + direction_arrows
            + obstacle_patches
            + [leader_plot, leader_traj]
        )

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=dt * 1000,
        blit=True
    )

    print(f"Generating {filename}...")

    ani.save(
        filename,
        writer='ffmpeg',
        fps=int(1 / dt)
    )

    plt.close(fig)

    print("Video generation complete.")
    

def plot_minimum_safety_distance(
        agents,
        d_min,
        dt,
        obstacles=None,
        show_obstacles=True,
        title="Minimum safety distance over time"):

    N = len(agents)
    histories = [np.asarray(ag.history) for ag in agents]

    T_steps = min(len(h) for h in histories)
    time = np.arange(T_steps) * dt

    min_agent_dist = np.zeros(T_steps)
    closest_pair = []

    for k in range(T_steps):
        min_dist_k = np.inf
        pair_k = None

        for i in range(N):
            p_i = histories[i][k, :2]

            for j in range(i + 1, N):
                p_j = histories[j][k, :2]

                dist_ij = np.linalg.norm(p_i - p_j)

                if dist_ij < min_dist_k:
                    min_dist_k = dist_ij
                    pair_k = (agents[i].id, agents[j].id)

        min_agent_dist[k] = min_dist_k
        closest_pair.append(pair_k)

    plt.figure(figsize=(10, 5))

    plt.plot(
        time,
        min_agent_dist,
        linewidth=2,
        label="Minimum agent-agent distance"
    )

    # Optional: obstacle clearance
    if obstacles is not None and show_obstacles:
        min_obs_clearance = np.zeros(T_steps)

        for k in range(T_steps):
            min_clearance_k = np.inf

            for i in range(N):
                p_i = histories[i][k, :2]

                for obs in obstacles:
                    p_o = np.asarray(obs["center"], dtype=float).reshape(-1)[:2]
                    r_o = float(obs["radius"])

                    # clearance relative to obstacle boundary
                    clearance = np.linalg.norm(p_i - p_o) - r_o

                    if clearance < min_clearance_k:
                        min_clearance_k = clearance

            min_obs_clearance[k] = min_clearance_k

        plt.plot(
            time,
            min_obs_clearance,
            linewidth=2,
            label="Minimum agent-obstacle border"
        )

        plt.axhline(
        y=d_min,
        linestyle="--",
        linewidth=2,
        label=f"d_min = {d_min}"
    )
        
        # plt.axhline(
        #     y=d_min,
        #     linestyle=":",
        #     linewidth=2,
        #     label=f"Obstacle border threshold = {d_min}",
        #     color = "red"
        # )

    plt.ylim(0.2, 2)
    plt.xlabel("Time (s)")
    plt.ylabel("Distance")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    idx_min = int(np.argmin(min_agent_dist))

    print("Minimum agent-agent distance:")
    print(f"  value = {min_agent_dist[idx_min]:.4f}")
    print(f"  time  = {time[idx_min]:.2f} s")
    print(f"  pair  = {closest_pair[idx_min]}")

    if obstacles is not None and show_obstacles:
        idx_min_obs = int(np.argmin(min_obs_clearance))

        print("\nMinimum agent-obstacle clearance:")
        print(f"  value = {min_obs_clearance[idx_min_obs]:.4f}")
        print(f"  time  = {time[idx_min_obs]:.2f} s")