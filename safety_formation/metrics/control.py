import numpy as np


def control_effort(controls, dt=1.0):
    controls = np.asarray(controls, dtype=float)
    return float(np.sum(np.linalg.norm(controls, axis=-1) ** 2) * dt)


def max_control_norm(controls):
    controls = np.asarray(controls, dtype=float)
    return float(np.max(np.linalg.norm(controls, axis=-1)))
