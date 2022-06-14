from functools import reduce

import numpy as np
import scipy.sparse

import tensorflow as tf

from core.decoder.vendor.network import mat_transpose
from core.utils import create_constant


def mat_mul(tensors):
    return tf.keras.backend.dot(tensors[0], tensors[1])


def filler_input_subgraph(fillers_shapes, shift_layer):
    subtree_as_inputs = []
    for shape in fillers_shapes:
        i = tf.keras.layers.Input(shape=(*shape[1:],))
        subtree_as_inputs.append(i)
    reshape_zero_level = tf.keras.layers.Reshape((1, 1, *fillers_shapes[0]))(subtree_as_inputs[0])
    reshape_first_level = tf.keras.layers.Reshape((1, *fillers_shapes[1]))(subtree_as_inputs[1])
    inputs_before_flattens = [
        reshape_zero_level,
        reshape_first_level,
        *subtree_as_inputs[2:]
    ]
    flatten_layers = [tf.keras.layers.Flatten()(input_layer) for input_layer in inputs_before_flattens]
    concat_layer = tf.keras.layers.concatenate(flatten_layers)
    transpose_layer = tf.keras.layers.Lambda(mat_transpose)(concat_layer)

    def mat_mul_with_constant(tensor):
        constant = tf.keras.backend.constant(shift_layer, dtype='float32')
        return tf.keras.backend.dot(constant, tensor)

    return subtree_as_inputs, tf.keras.layers.Lambda(mat_mul_with_constant)(transpose_layer)


def shift_matrix(role, filler_size, max_depth, name, mode='dense'):
    """
    Builds the W_{cons0} matrix
    :param max_depth: maximum depth of the resulting tree
    :return: W_{cons0} matrix
    """
    # print('Building {} matrix'.format(name))
    role_len = role.shape[0]

    # constructing I (identity matrix of the given depth)
    num_cols = reduce(lambda acc, depth: acc + (filler_size * (role_len ** depth)), range(max_depth), 0)
    num_rows = num_cols * role_len  # + 1 no row for magic epsilon (p. 315) - is it a root?

    if mode == 'dense':
        res_matrix = np.zeros((num_rows, num_cols))
    elif mode == 'sparse':
        res_matrix = scipy.sparse.lil_matrix((num_rows, num_cols))
    else:
        raise NotImplementedError(f'Given mode {mode} is not supported for shift_matrix() method')

    for row_index, col_index in zip(range(0, num_rows - role_len + 1, role_len), range(num_cols)):
        # broadcast will not work for lil_matrix use case
        for role_component_index, role_component in enumerate(role):
            res_matrix[row_index + role_component_index, col_index] = role_component
    return res_matrix


def build_join_branch(roles, fillers_shapes):
    filler_len = fillers_shapes[0][1]
    max_depth = len(fillers_shapes)

    variable_inputs = []
    matmul_layers = []
    for role_index, role in enumerate(roles):
        shift_input = create_constant(role, filler_len, max_depth, f'constant_input_cons{role_index}_', shift_matrix)

        inputs, matmul_layer = filler_input_subgraph(fillers_shapes, shift_input)
        variable_inputs.append(inputs)
        matmul_layers.append(matmul_layer)

    sum_layer = tf.keras.layers.Add()(matmul_layers)

    return (
               tuple(variable_inputs)
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
    variable_inputs, output = build_join_branch(roles, fillers_shapes)

    model_inputs = []
    for inputs_list in variable_inputs:
        model_inputs.extend(inputs_list)

    return tf.keras.Model(
        inputs=model_inputs,
        outputs=output
    )
