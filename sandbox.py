import numpy as np

from math_utils import kronecker_delta, feed_forward_recursion_matrix, multiplied_identity, \
    single_dimension_transfer_weights

if __name__ == '__main__':
    roles_basis = np.array([
        [10, -3],  # r_{0}
        [3, 10]  # r_{1}
    ])

    assert np.sum(np.prod(roles_basis, axis=0)) == 0, 'Roles vectors should be orthogonal'

    filler_basis = np.array([
        [2, 0, 1],  # A
        [4, 10, 0],  # B
        [0, 3, 5]  # C
    ])

    assert np.sum(np.prod(filler_basis, axis=0)) == 0, 'Filler vectors should be orthogonal'

    # A (x) r_{0} + [B (x) r_{0} + C (x) r_{1}] (x) r_{1}

    # matrix multiplication

    # A = [ 2 3 4
    #       5 6 7 ]

    # B = [ 5 6
    #       7 8
    #       9 1 ]

    # A * B = [ 10+21+36 12+24+4    = [67 40
    #           25+42+63 30+48+7]      130 85]

    A = np.array([[2, 3, 4], [5, 6, 7]])
    B = np.array([[5, 6], [7, 8], [9, 1]])
    print(A.dot(B))

    # encoding the structure, 3 is the maximum depth
    #               epsilon
    #           /                   \
    #       A                   /       \
    #                          B        C
    MAXIMUM_TREE_DEPTH = 3
    W = np.zeros((MAXIMUM_TREE_DEPTH, MAXIMUM_TREE_DEPTH))

    res = multiplied_identity(3, MAXIMUM_TREE_DEPTH)

    # 1. Make A (x) r_0

    # 1.1. Prepare weights

    # 1.1.1 Prepare translation from level 2 to level 3
    res = single_dimension_transfer_weights(level=1,
                                            roles_dim=roles_basis[0].shape[0],
                                            filler_dim=filler_basis[0].shape[0],
                                            role_m=roles_basis[0])

    feed_forward_recursion_matrix(MAXIMUM_TREE_DEPTH, roles_basis[0].shape[0])
