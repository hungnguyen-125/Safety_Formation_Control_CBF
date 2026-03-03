import numpy as np
import matplotlib
matplotlib.use("TkAgg")   # hoặc "Qt5Agg"
import matplotlib.pyplot as plt

from safety_formation.control_law.cbf import CentralizedCBF

def main():
    # --- Cấu hình mô phỏng ---
    N = 3
    R_circle = 150
    dt = 0.1
    T_max = 100
    gamma = 0.1
    d_min = 9.0
    k1, k2 = 1.0, 2.0

    # --- Khởi tạo trạng thái ---
    agents_list = []

    class AgentObj:
        def __init__(self, id, pos, target):
            self.id = id
            self.pos = pos
            self.vel = np.zeros(2)
            self.target = target
            self.alpha = 20.0
            self.gamma = gamma

    for i in range(N):
        angle = 2 * np.pi * i / N
        pos = np.array([R_circle * np.cos(angle), R_circle * np.sin(angle)], dtype=float)
        target = -pos - 10.0
        agents_list.append(AgentObj(i, pos, target))

    # Fully connected topology
    class FullTopology:
        adj_matrix = np.ones((N, N)) - np.eye(N)

    if CentralizedCBF is None:
        raise ImportError(
            "Bạn chưa import CentralizedCBF. Hãy sửa phần import ở đầu file."
        )

    cbf_filter = CentralizedCBF(gamma=gamma, safety_dis=d_min)

    # ================== REALTIME PLOT ==================
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-R_circle - 80, R_circle + 80)
    ax.set_ylim(-R_circle - 80, R_circle + 80)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title("Robot Position Swapping with Centralized CBF (Realtime)")

    # Vẽ target
    for ag in agents_list:
        ax.scatter(ag.target[0], ag.target[1], marker="x", color="red")

    # Artist cho agent + trail
    points = [ax.plot([], [], "o")[0] for _ in range(N)]
    trails = [ax.plot([], [], alpha=0.3)[0] for _ in range(N)]
    history = [[] for _ in range(N)]

    # Text realtime
    info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

    # Vẽ frame đầu tiên để “show”
    fig.canvas.draw()
    fig.canvas.flush_events()

    # --- Vòng lặp mô phỏng ---
    for step in range(int(T_max / dt)):

        # Nếu user đóng cửa sổ thì dừng loop (tránh treo)
        if not plt.fignum_exists(fig.number):
            break

        # 1) u_nom
        u_nom_all = []
        for ag in agents_list:
            u_i = -k1 * (ag.pos - ag.target) - k2 * ag.vel
            u_nom_all.append(u_i)
        u_nom_all = np.asarray(u_nom_all, dtype=float)

        # 2) CBF filter
        u_safe_all = cbf_filter.compute_safe_control(agents_list, FullTopology(), u_nom_all)
        if u_safe_all is None:
            print("CBF returned None -> stop")
            break
        u_safe_all = np.asarray(u_safe_all, dtype=float)

        # 3) update physics + plot
        for i, ag in enumerate(agents_list):
            ag.vel = ag.vel + u_safe_all[i] * dt
            ag.pos = ag.pos + ag.vel * dt

            history[i].append(ag.pos.copy())
            traj = np.asarray(history[i])

            points[i].set_data([ag.pos[0]], [ag.pos[1]])
            trails[i].set_data(traj[:, 0], traj[:, 1])

        # 4) realtime info (min distance)
        min_d = float("inf")
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(agents_list[i].pos - agents_list[j].pos)
                if d < min_d:
                    min_d = d
        info_text.set_text(f"t = {step*dt:.2f}s\nmin_dist = {min_d:.2f}")

        # 5) refresh
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(dt)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()