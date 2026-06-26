import numpy as np


def pairwise_distances(positions):
    positions = np.asarray(positions, dtype=float)
    n_agents = positions.shape[0]
    distances = []

    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            distances.append(np.linalg.norm(positions[i] - positions[j]))

    return np.asarray(distances, dtype=float)


def minimum_pairwise_distance(positions_over_time):
    positions_over_time = np.asarray(positions_over_time, dtype=float)
    min_distance = np.inf

    for positions in positions_over_time:
        distances = pairwise_distances(positions)
        if distances.size > 0:
            min_distance = min(min_distance, float(np.min(distances)))

    return min_distance


def safety_violation_count(positions_over_time, safety_distance):
    positions_over_time = np.asarray(positions_over_time, dtype=float)
    count = 0

    for positions in positions_over_time:
        distances = pairwise_distances(positions)
        count += int(np.sum(distances < safety_distance))

    return count
