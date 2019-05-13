from functools import reduce

import numpy as np

from recursive_structure import pre_process_roles, calculate_tensor_representation_shape_as_2d_matrix
from src.decoder.vendor.network import build_filler_decoder_network as keras_filler_decoder_network, \
    prepare_shapes as prepare_decoder_shapes
from src.joiner.vendor.network import build_tree_joiner_network


# def make_left_child_from(another_tree, shift_left_matrix):
#     print('Making left child from a tree')
#     max_size = shift_left_matrix.shape[1]
#     flattened_tree = another_tree.flatten()
#     flattened_tree.resize((max_size,), refcheck=False)
#     return np.dot(shift_left_matrix, flattened_tree)


# def build_shift_matrix(role, max_depth):
#     """
#     Builds the W_{cons0} matrix
#     :param max_depth: maximum depth of the resulting tree
#     :return: W_{cons0} matrix
#     """
#     print('Building W_{cons0} matrix')
#     # constructing I (identity matrix of the given depth)
#     num_cols = reduce(lambda x, y: x + 2 ** y, range(max_depth), 0)
#     num_rows = num_cols * len(role) + 1 # row for magic epsilon (p. 315) - is it a root?
#     res_matrix = np.zeros((num_rows, num_cols))
#     for row_index, col_index in zip(range(1, num_rows - len(role) + 1, len(role)), range(num_cols)):
#         res_matrix[row_index:row_index + len(role), col_index] = role
#     return res_matrix


def generate_shapes(max_tree_depth, role_shape, filler_shape):
    shapes = []
    for i in range(max_tree_depth):
        roles_shape_addon = [role_shape[0] for j in range(i)]
        shapes.append(np.array([filler_shape[0], *roles_shape_addon]))
    return np.array(shapes)


if __name__ == '__main__':
    """
    First use case for the structure that should be shifted left

    Starting from:
    root
    
    We want to get:
    root
    |
    A (left-child-of-root)
    """
    MAX_TREE_DEPTH = 3
    fillers_case_1 = np.array([
        [8, 0, 0, 0],  # A
        [0, 15, 0, 0],  # B
        [0, 0, 10, 0],  # C
        [0, 0, 0, 3],  # D
    ])
    roles_case_1 = np.array([
        [10, 0],  # r_0
        [0, 5],  # r_1
    ])
    mapping_case_2 = {
        'A': [0, 0]
    }
    order_case_2 = ['A', ]
    dual_basic_roles_case_1 = np.linalg.inv(roles_case_1)
    final_dual_roles = pre_process_roles(dual_basic_roles_case_1, mapping_case_2, order_case_2)

    fillers_shapes = generate_shapes(max_tree_depth=MAX_TREE_DEPTH,
                                     role_shape=roles_case_1[0].shape,
                                     filler_shape=fillers_case_1[0].shape)
    keras_joiner = build_tree_joiner_network(roles_shape=roles_case_1.shape, fillers_shapes=fillers_shapes)
    # fillers_joined = keras_joiner.predict_on_batch([
    #     reshaped_tensor_representation,
    #     reshaped_dual_roles
    # ])

    # shift_left_matrix = build_shift_matrix(roles_case_1[0], MAX_TREE_DEPTH)
    # new_tree = make_left_child_from(fillers_case_1[0], shift_left_matrix)
    # new_tree = make_left_child_from(new_tree, shift_left_matrix)
    # print('done with building a tree')
    # print(new_tree)
    #
    # number_fillers, dim_fillers = fillers_case_1.shape
    # number_basic_roles, dim_roles = roles_case_1.shape
    #
    # tr_shape = calculate_tensor_representation_shape_as_2d_matrix(dim_fillers, dim_roles, MAX_TREE_DEPTH)
    #
    # print('Building Keras decoder')
    # dual_roles_shape = prepare_decoder_shapes(mapping=mapping_case_2,
    #                                           dual_roles=final_dual_roles)
    # keras_decoder = keras_filler_decoder_network(input_shapes=(new_tree.shape, dual_roles_shape))
    #
    # print('Running Keras decoder')
    # reshaped_dual_roles = final_dual_roles.reshape(dual_roles_shape)
    #
    # reshaped_tensor_representation = new_tree.reshape(tr_shape)
    #
    # fillers_restored = keras_decoder.predict_on_batch([
    #     reshaped_tensor_representation,
    #     reshaped_dual_roles
    # ])
    #
    # print('done!')

    """
    Second use case for the structure that should be shifted left

    Starting from:
    root
    |
    A (left-child-of-root)

    We want to get:
    root
    |
    |
    A (left-child-of-left-child-of-root)
    """
