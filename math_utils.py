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
    res = np.identity(dim)
    for block_idx in range(times):
        res = matrix_multiplication(res, np.identity(dim))
    return res


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

    :param matrix_a: first matrix
    :param matrix_b: second matrix
    :return: multiplication result
    """
    return np.tensordot(matrix_a, matrix_b, axes=0)
    # return np.kron(matrix_a, matrix_b)


def single_dimension_transfer_weights(level, roles_dim, filler_dim, role_m):
    filler_m = np.identity(filler_dim)

    identities = multiplied_identity(level, roles_dim)
    # TODO: eliminate this hard dependency on a role to be a 2-component vector
    # role_m_t = np.array([[role_m[0]], [role_m[1]]])

    filler_role = matrix_multiplication(filler_m, role_m)

    return matrix_multiplication(identities, filler_role)
