import numpy as np
from safety_formation.formation.topology import Topology

def main():
    A = np.array([
        [0, 1, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0],
    ], dtype=float)

    leader_access = [1, 0, 1, 0, 0, 0]

    topo = Topology(num_agents=6, adjacency_matrix=A, leader_access=leader_access)

    # --- Expected matrices ---
    L_expected = np.array([
        [ 2, -1,  0,  0,  0, -1],
        [-1,  2, -1,  0,  0,  0],
        [ 0, -1,  2, -1,  0,  0],
        [ 0,  0, -1,  2, -1,  0],
        [ 0,  0,  0, -1,  2, -1],
        [-1,  0,  0,  0, -1,  2],
    ], dtype=float)

    D_lead_expected = np.diag([1, 0, 1, 0, 0, 0]).astype(float)
    H_expected = L_expected + D_lead_expected

    # --- Assertions ---
    assert topo.adj_matrix.shape == (6, 6)
    assert topo.D_lead.shape == (6, 6)

    assert np.allclose(topo.laplacian_matrix, L_expected)
    assert np.allclose(topo.D_lead, D_lead_expected)
    assert np.allclose(topo.get_augmented_laplacian(), H_expected)

    # Neighbor tests (1-based IDs)
    assert set(topo.get_neighbors(1).tolist()) == {2, 6}
    assert set(topo.get_neighbors(3).tolist()) == {2, 4}
    assert set(topo.get_neighbors(6).tolist()) == {1, 5}

    # Leader access tests
    assert topo.sees_leader(1) is True
    assert topo.sees_leader(2) is False
    assert topo.sees_leader(3) is True

    print("Topology tests passed.")


if __name__ == "__main__":
    main()