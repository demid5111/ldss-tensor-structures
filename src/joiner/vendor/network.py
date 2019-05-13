from functools import reduce

import numpy as np

from keras.engine import Input
from keras.layers import Lambda, Flatten, Add, concatenate
from keras.models import Model

# from src.decoder.vendor.network import mat_mul

from keras import backend as K
from keras.layers import Layer

from src.decoder.vendor.network import mat_transpose


class ShiftMatrixCreatorLayer(Layer):
    def __init__(self, max_depth, filler_size, **kwargs):
        self.max_depth = max_depth
        self.filler_size = filler_size

        super(ShiftMatrixCreatorLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[0], 15),
                                      initializer='uniform',
                                      trainable=True)
        super(ShiftMatrixCreatorLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        # assert isinstance(x, list)
        # a, b = x
        # return [K.dot(a, self.kernel) + b, K.mean(b, axis=-1)]
        return x

    def compute_output_shape(self, input_shape):
        # assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        role_len = input_shape[0]
        num_cols = reduce(lambda acc, depth: acc + (self.filler_size * (2 ** depth)), range(self.max_depth), 0)
        # num_cols = 0
        # for depth in range(self.max_depth):
        #     num_cols += (self.filler_size * (2 ** depth))
        num_rows = num_cols * role_len  # + 1 no row for magic epsilon (p. 315) - is it a root?
        return [(num_rows, num_cols)]


def mat_mul(tensors):
    return K.dot(tensors[0], tensors[1])


def filler_input_subgraph(fillers_shapes, shift_layer):
    subtree_as_inputs = []
    for shape in fillers_shapes:
        if len(shape) < 2:
            bath_shape = (1, 1, *shape)
        else:
            bath_shape = (1, *shape)
        i = Input(shape=shape, batch_shape=bath_shape)
        subtree_as_inputs.append(i)
    flatten_layers = [Flatten()(input_layer) for input_layer in subtree_as_inputs]
    concat_layer = concatenate(flatten_layers)
    transpose_layer = Lambda(mat_transpose)(concat_layer)
    return subtree_as_inputs, Lambda(mat_mul)([
        shift_layer,
        transpose_layer
    ])


# def shift_matrix(role, max_depth):
#     """
#     Builds the W_{cons0} matrix
#     :param max_depth: maximum depth of the resulting tree
#     :return: W_{cons0} matrix
#     """
#     print('Building W_{cons0} matrix')
#     role_len = role.shape[0].value
#
#     # constructing I (identity matrix of the given depth)
#     num_cols = reduce(lambda x, y: x + 2 ** y, range(max_depth+1), 0)
#     num_rows = num_cols * role_len  # + 1 no row for magic epsilon (p. 315) - is it a root?
#     res_matrix = np.identity(num_cols)
#     multiplied_matrix = np.multiply(res_matrix, role)
#     # multiplied_matrix = res_matrix * role
#     # return multiplied_matrix.reshape((num_rows, num_cols))
#     # return multiplied_matrix
#
#     # res_matrix = np.zeros((num_rows, num_cols))
#     # for row_index, col_index in zip(range(1, num_rows - role_len + 1, role_len), range(num_cols)):
#     #     res_matrix[row_index:row_index + role_len, col_index] = role
#     # return res_matrix
#
#     ones = np.zeros((num_cols,num_cols,role_len,1))
#     for i_row, rows in enumerate(ones):
#         for i_col, cols in enumerate(rows):
#             if i_col != i_row:
#                 continue
#             ones[i_row][i_col] = np.ones((role_len, 1))
#
#     return K.T.tensordot(res_matrix, role, axes=0)
#     # return ones * multiplied_matrix



def build_tree_joiner_network(roles_shape, fillers_shapes):
    """
    Building the following network:
    roles  input(level 0) input(level 1) input(level 2) ... input(level 0) input(level 1) input(level 2)
    |       |               |               |                   |           |               |
    matrix  flatten         flatten         flatten         flatten         flatten         flatten
    creator \                 |             |                   \           |               |
    (cons0) \                 |             |                    \          |               |
    \                concatenate                                        concatenate
    \                   |                  (matrix creator cons1)\                   |
            matmul                                                      matmul
                \                                                   /
                                    sum

    :param roles_shape:
    :param fillers_shapes:
    :return:
    """
    num_roles, role_len = roles_shape
    filler_len = fillers_shapes[0][0]
    num_roles = roles_shape[0]
    max_depth = len(fillers_shapes)

    role_input = Input(shape=roles_shape[1:],
                       batch_shape=roles_shape)
    role_left, role_right = Lambda(lambda x: K.tf.split(x, num_or_size_splits=num_roles, axis=0))(role_input)
    print('0:', role_left.shape)  # 0: (?, 1024, 1)
    print('1:', role_right.shape)  # 1: (?, 1024, 1)

    shift_left_layer = ShiftMatrixCreatorLayer(max_depth, filler_len)(role_left)
    left_inputs, left_matmul_layer = filler_input_subgraph(fillers_shapes, shift_left_layer)

    shift_right_layer = ShiftMatrixCreatorLayer(max_depth, filler_len)(role_right)
    right_inputs, right_matmul_layer = filler_input_subgraph(fillers_shapes, shift_right_layer)

    sum_layer = Add()([
        left_matmul_layer,
        right_matmul_layer
    ])

    return Model(
        inputs=[
            role_input,
            *left_inputs,
            *right_inputs
        ],
        outputs=sum_layer)
