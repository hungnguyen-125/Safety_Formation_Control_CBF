import numpy as np

from safety_formation import Agent
from safety_formation.simulation import Scenario, SimulationRunner


def constant_policy(step, time, agents, topology):
    return np.array([
        [1.0, 0.0],
        [0.0, -1.0],
    ])


def main():
    agents = [
        Agent(
            agent_id=1,
            x0=[0.0, 0.0, 0.0, 0.0],
            f_target=[0.0, 0.0, 0.0, 0.0],
        ),
        Agent(
            agent_id=2,
            x0=[1.0, 0.0, 0.0, 0.0],
            f_target=[0.0, 0.0, 0.0, 0.0],
        ),
    ]

    scenario = Scenario(
        agents=agents,
        dt=0.1,
        steps=20,
        control_policy=constant_policy,
    )

    result = SimulationRunner(scenario).run()
    print(result.final_state)


if __name__ == "__main__":
    main()
