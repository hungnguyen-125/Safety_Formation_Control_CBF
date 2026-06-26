import numpy as np


def formation_errors(states, formation_offsets, leader_state=None):
    states = np.asarray(states, dtype=float)
    formation_offsets = np.asarray(formation_offsets, dtype=float)

    if leader_state is None:
        leader_state = np.zeros(states.shape[-1], dtype=float)
    else:
        leader_state = np.asarray(leader_state, dtype=float).reshape(states.shape[-1])

    return states - formation_offsets - leader_state


def formation_error_norms(states, formation_offsets, leader_state=None):
    errors = formation_errors(states, formation_offsets, leader_state=leader_state)
    return np.linalg.norm(errors, axis=-1)


def final_formation_error(states, formation_offsets, leader_state=None):
    norms = formation_error_norms(states, formation_offsets, leader_state=leader_state)
    return norms[-1]
