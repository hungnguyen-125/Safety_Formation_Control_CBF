from dataclasses import dataclass

import numpy as np


@dataclass
class SimulationResult:
    times: np.ndarray
    states: np.ndarray
    controls: np.ndarray

    @property
    def final_state(self):
        return self.states[-1]

    @property
    def positions(self):
        return self.states[:, :, 0:2]
