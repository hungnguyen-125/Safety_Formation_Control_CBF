import numpy as np
from safety_formation.formation.agent import Agent

def main():
    dt = 0.1
    T = 2.0
    steps = int(T / dt)

    # x0 = [x, y, vx, vy]
    x0 = [0.0, 0.0, 0.0, 0.0]
    # formation offset target [fx, fy, 0, 0]
    f_target = [1.0, 0.5, 0.0, 0.0]

    agent = Agent(agent_id=1, x0=x0, f_target=f_target)

    # control input u = [ax, ay]
    u = [1.0, -0.5]

    for _ in range(steps):
        agent.update_physics(u=u, dt=dt)

    hist = agent.get_full_history()  # shape (steps, 4)
    print("Final state [x, y, vx, vy]:", hist[-1])
    print("Final pos:", agent.pos.flatten())
    print("Final vel:", agent.vel.flatten())

    # Test (assert)
    assert hist.shape == (steps, 4)
    assert agent.pos.shape == (2, 1)
    assert agent.vel.shape == (2, 1)

if __name__ == "__main__":
    main()