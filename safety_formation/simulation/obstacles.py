from dataclasses import dataclass

import numpy as np


@dataclass
class CircularObstacle:
    center: object
    radius: float

    def as_dict(self):
        return {
            "center": np.asarray(self.center, dtype=float),
            "radius": float(self.radius),
        }
