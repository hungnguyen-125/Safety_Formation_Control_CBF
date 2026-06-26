import numpy as np

from safety_formation.metrics import (
    control_effort,
    final_formation_error,
    formation_error_norms,
    max_control_norm,
    minimum_pairwise_distance,
    pairwise_distances,
    safety_violation_count,
)


def test_safety_metrics_compute_pairwise_distances_and_violations():
    positions_over_time = np.array([
        [[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]],
        [[0.0, 0.0], [0.4, 0.0], [0.0, 2.0]],
    ])

    assert np.allclose(pairwise_distances(positions_over_time[0]), [2.0, 3.0, np.sqrt(13.0)])
    assert minimum_pairwise_distance(positions_over_time) == 0.4
    assert safety_violation_count(positions_over_time, safety_distance=0.5) == 1


def test_formation_metrics_compute_error_norms():
    states = np.array([
        [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        [[0.5, 0.0, 0.0, 0.0], [1.5, 0.0, 0.0, 0.0]],
    ])
    offsets = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ])

    norms = formation_error_norms(states, offsets)

    assert np.allclose(norms, [[0.0, 0.0], [0.5, 0.5]])
    assert np.allclose(final_formation_error(states, offsets), [0.5, 0.5])


def test_control_metrics_compute_effort_and_max_norm():
    controls = np.array([
        [[1.0, 0.0], [0.0, 2.0]],
        [[0.0, 0.0], [3.0, 4.0]],
    ])

    assert control_effort(controls, dt=0.1) == 3.0
    assert max_control_norm(controls) == 5.0
