import numpy as np


def kronecker_delta(i, j):
    return 1 if i == j else 0


def feed_forward_recursion_matrix(k, dim):
    m = None

    res = 1
    identity_ref = np.identity(dim)

    for i in range(1, k + 1):
        intermediate_res = identity_ref
        for block_idx in range(1, i):
            intermediate_res = np.tensordot(intermediate_res, identity_ref, axes=0)

        res = 1 + intermediate_res

    return res


def multiplied_identity(times, dim):
    """
    Calculates 1_{R} (x) 1_{R} (x) ... (x) 1_{R}

    Not clear if real multiplication should happen, if so:

    >> res = np.identity(dim)
    >> for block_idx in range(times):
    >>     res = matrix_multiplication(res, np.identity(dim))
    >> return res

    :param times: not clear if it is used
    :param dim: number of elements in the role vector
    :return: identity matrix of size times x times (p. 312)
    """
    # TODO: define what is the dimension of 1^{\otimes d}
    # why is it x2 for each lower step on (p. 312)
    return np.identity(times)


def matrix_multiplication(matrix_a, matrix_b):
    """
    The reason for the function is simple - as tensordot and kron are conceptually
    the same and we want to have an option to quickly switch between them,
    this function is the entry point.

    >> A = np.array([1, 2])
    >> B = np.array([3, 4])

    Result is matrix (dimension is 1):
    >> C = np.tensordot(A, B, axes=0)
    >> print(C)
    >> [1*[3,4], 2*[3,4]] = [[3,4], [6,8]]

    It seems that np.tensordot(A,B, axes=0) and np.kron(A,B) are doing the same
    except the fact that tensordot creates the true blocked structure
    kron creates the single matrix with the same kuazi-block structure (technically
    this is just a k*k matrix)

    >> c_kron = np.kron(A, B)

    Final decision is to use np.tensordot as we need to keep block structure
    of intermediate computation results

    :param matrix_a: first matrix
    :param matrix_b: second matrix
    :return: multiplication result
    """
    # return np.kron(matrix_a, matrix_b)
    return np.tensordot(matrix_a, matrix_b, axes=0)


def single_dimension_transfer_weights(level, filler_dim, role_v):
    """
    Calculates weights for applying the role to the given filler that results in shifting it one
    level down

    :param level: target level (where the filler should appear after the role)
    :param filler_dim: length of filler vectors
    :param role_v: vector of the role that is applied to the filler
    :return: weights matrix
    """
    filler_m = np.identity(filler_dim)

    # (p. 313)
    # W_{cons0} = I (x) 1_{A} (x) r_{0} is equivalent to
    # W_{cons0} = 1_{A} (x) I (x) r_{0}
    identities = multiplied_identity(level, role_v.shape[0])

    # transposing is needed, according to the (p. 316)
    # np.transpose does not work as expected for 1D arrays
    role_m_t = role_v.reshape(role_v.shape[0], 1)

    # firstly do I (x) r_{0}
    identity_role = matrix_multiplication(identities, role_m_t)

    # secondly do 1_{A} (x) (I (x) r_{0})
    return matrix_multiplication(filler_m, identity_role)


if __name__ == '__main__':
    # roles_basis = np.array([
    #     [10, -3],  # r_{0}
    #     [3, 10]  # r_{1}
    # ])
    #
    # assert np.sum(np.prod(roles_basis, axis=0)) == 0, 'Roles vectors should be orthogonal'
    #
    # filler_basis = np.array([
    #     [2, 0, 1],  # A
    #     [4, 10, 0],  # B
    #     [0, 3, 5]  # C
    # ])
    #
    # assert np.sum(np.prod(filler_basis, axis=0)) == 0, 'Filler vectors should be orthogonal'
    #
    # # A (x) r_{0} + [B (x) r_{0} + C (x) r_{1}] (x) r_{1}
    #
    # # encoding the structure, 3 is the maximum depth
    # #               epsilon
    # #           /                   \
    # #       A                   /       \
    # #                          B        C
    # MAXIMUM_TREE_DEPTH = 3
    # W = np.zeros((MAXIMUM_TREE_DEPTH, MAXIMUM_TREE_DEPTH))
    #
    # # 1. Make A (x) r_0
    #
    # # 1.1. Prepare weights
    #
    # # 1.1.1 Prepare translation from level 2 to level 3
    # res = single_dimension_transfer_weights(level=1,
    #                                         filler_dim=filler_basis[0].shape[0],
    #                                         role_v=roles_basis[0])
    #
    # feed_forward_recursion_matrix(MAXIMUM_TREE_DEPTH, roles_basis[0].shape[0])
    pass
