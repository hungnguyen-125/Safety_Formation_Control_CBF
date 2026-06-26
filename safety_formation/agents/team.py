import numpy as np


class Team:
    """Container for batch operations on a group of agents."""

    def __init__(self, agents):
        self.agents = list(agents)

    def __iter__(self):
        return iter(self.agents)

    def __len__(self):
        return len(self.agents)

    def __getitem__(self, index):
        return self.agents[index]

    @property
    def states(self):
        return np.vstack([agent.state for agent in self.agents])

    @property
    def formation_offsets(self):
        return np.vstack([agent.f for agent in self.agents])

    @property
    def positions(self):
        return np.vstack([agent.pos.reshape(1, -1) for agent in self.agents])

    @property
    def velocities(self):
        return np.vstack([agent.vel.reshape(1, -1) for agent in self.agents])

    def update_physics(self, controls, dt):
        controls = np.asarray(controls, dtype=float)

        if controls.shape[0] != len(self.agents):
            raise ValueError(
                f"controls must provide one row per agent, got {controls.shape[0]} "
                f"for {len(self.agents)} agents"
            )

        for agent, control in zip(self.agents, controls):
            agent.update_physics(control, dt)

    def save_history(self):
        for agent in self.agents:
            agent.save_history()

    def clear_history(self):
        for agent in self.agents:
            agent.clear_history()

    def get_histories(self):
        return [agent.get_full_history() for agent in self.agents]
