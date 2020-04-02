from functools import reduce

import numpy as np
from keras import backend as K
from keras.engine import Input
from keras.layers import Lambda, Flatten, Add, concatenate, Reshape
from keras.models import Model

from core.decoder.vendor.network import mat_transpose


def mat_mul(tensors):
    return K.dot(tensors[0], tensors[1])


def filler_input_subgraph(fillers_shapes, shift_layer):
    subtree_as_inputs = []
    for shape in fillers_shapes:
        i = Input(shape=(*shape[1:],), batch_shape=(*shape,))
        subtree_as_inputs.append(i)
    reshape_zero_level = Reshape((1, 1, *fillers_shapes[0]))(subtree_as_inputs[0])
    reshape_first_level = Reshape((1, *fillers_shapes[1]))(subtree_as_inputs[1])
    inputs_before_flattens = [
        reshape_zero_level,
        reshape_first_level,
        *subtree_as_inputs[2:]
    ]
    flatten_layers = [Flatten()(input_layer) for input_layer in inputs_before_flattens]
    concat_layer = concatenate(flatten_layers)
    transpose_layer = Lambda(mat_transpose)(concat_layer)
    return subtree_as_inputs, Lambda(mat_mul)([
        shift_layer,
        transpose_layer
    ])


def constant_input(role, filler_len, max_depth, name, matrix_creator):
    np_constant = matrix_creator(role, filler_len, max_depth, name)
    tf_constant = K.constant(np_constant)
    return Input(tensor=tf_constant, batch_shape=np_constant.shape, shape=np_constant.shape, dtype='int32', name=name)


def shift_matrix(role, filler_size, max_depth, name):
    """
    Builds the W_{cons0} matrix
    :param max_depth: maximum depth of the resulting tree
    :return: W_{cons0} matrix
    """
    print('Building {} matrix'.format(name))
    role_len = role.shape[0]

    # constructing I (identity matrix of the given depth)
    num_cols = reduce(lambda acc, depth: acc + (filler_size * (2 ** depth)), range(max_depth), 0)
    num_rows = num_cols * role_len  # + 1 no row for magic epsilon (p. 315) - is it a root?

    res_matrix = np.zeros((num_rows, num_cols))
    for row_index, col_index in zip(range(0, num_rows - role_len + 1, role_len), range(num_cols)):
        res_matrix[row_index:row_index + role_len, col_index] = role
    return res_matrix


def build_join_branch(roles, fillers_shapes):
    filler_len = fillers_shapes[0][1]
    max_depth = len(fillers_shapes)
    left_shift_input = constant_input(roles[0], filler_len, max_depth, 'constant_input_(cons0)', shift_matrix)
    left_inputs, left_matmul_layer = filler_input_subgraph(fillers_shapes, left_shift_input)

    right_shift_input = constant_input(roles[1], filler_len, max_depth, 'constant_input_(cons1)', shift_matrix)
    right_inputs, right_matmul_layer = filler_input_subgraph(fillers_shapes, right_shift_input)

    sum_layer = Add()([
        left_matmul_layer,
        right_matmul_layer
    ])

    return (
        (
            left_shift_input,
            right_shift_input
        ),
        (
            left_inputs,
            right_inputs
        )
    ), sum_layer


def build_tree_joiner_network(roles, fillers_shapes):
    """
    Building the following network.

    Draw it with instructions from README.md

    >>> from keras.utils import plot_model
    >>> keras_joiner = None # some place in the main code after this function is called
    >>> plot_model(keras_joiner, to_file='keras_joiner.png')

    :param fillers_shapes:
    :return:
    """

    inputs, output = build_join_branch(roles, fillers_shapes)
    const_inputs, variable_inputs = inputs
    left_inputs, right_inputs = variable_inputs

    return Model(
        inputs=[
            *const_inputs,
            *left_inputs,
            *right_inputs
        ],
        outputs=output)
