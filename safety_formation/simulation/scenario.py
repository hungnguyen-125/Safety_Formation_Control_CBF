from dataclasses import dataclass, field


@dataclass
class Scenario:
    agents: list
    dt: float
    steps: int
    topology: object = None
    control_policy: object = None
    nominal_controller: object = None
    safety_filter: object = None
    obstacles: list = field(default_factory=list)
    disturbances: list = field(default_factory=list)
    leader_state: object = None

    @classmethod
    def from_duration(cls, agents, dt, duration, **kwargs):
        return cls(
            agents=agents,
            dt=dt,
            steps=int(duration / dt),
            **kwargs,
        )
