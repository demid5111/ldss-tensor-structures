from functools import reduce

import numpy as np

from keras.models import Model

from core.joiner.vendor.network import constant_input, filler_input_subgraph


def unshift_matrix(role, filler_size, max_depth, name='unshift'):
    """
    Builds the W_{cons0} matrix
    :param name:
    :param max_depth: maximum depth of the resulting tree
    :return: W_{cons0} matrix
    """
    # print('Building {} matrix'.format(name))
    role_len = role.shape[0]

    # constructing I (identity matrix of the given depth)
    num_rows = reduce(lambda acc, depth: acc + (filler_size * (2 ** depth)), range(max_depth), 0)
    num_cols = num_rows * role_len

    res_matrix = np.zeros((num_rows, num_cols))
    for row_index, col_index in zip(range(num_rows), range(0, num_cols - role_len + 1, role_len)):
        res_matrix[row_index, col_index: col_index + role_len] = role
    return res_matrix


def build_tree_unshifter_network(roles, fillers_shapes, role_index=0):
    """
    Building the following network.

    Draw it with instructions from README.md

    >>> from keras.utils import plot_model
    >>> keras_joiner = None # some place in the main code after this function is called
    >>> plot_model(keras_joiner, to_file='keras_joiner.png')

    :param roles_shape:
    :param fillers_shapes:
    :return:
    """
    filler_len = fillers_shapes[0][1]
    max_depth = len(fillers_shapes)

    layer_name = 'constant_input_(ex0)'.format(role_index)
    left_shift_input = constant_input(roles[role_index], filler_len, max_depth, layer_name, unshift_matrix)
    left_inputs, left_matmul_layer = filler_input_subgraph(fillers_shapes, left_shift_input)

    return Model(
        inputs=[
            left_shift_input,
            *left_inputs,
        ],
        outputs=left_matmul_layer)
