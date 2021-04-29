import numpy as np
import tensorflow as tf

from demo.recursive_structure import pre_process_roles
from demo.shifting_structure import main as shifting_main, generate_shapes
from core.unshifter.vendor.network import build_tree_unshifter_network


def reshape_to_satisfy_max_depth_after_unshift(tensor_representation, max_tree_depth, role_shape, filler_shape):
    expected_shapes = generate_shapes(max_tree_depth=max_tree_depth + 1,
                                      role_shape=role_shape,
                                      filler_shape=filler_shape)

    res_representation = [None for _ in range(max_tree_depth + 1)]
    existing_levels = set()
    for level_representation in tensor_representation:
        for j, expected_shape in enumerate(expected_shapes):
            if np.array_equal(level_representation.shape, expected_shape):
                existing_levels.add(j)
                res_representation[j] = level_representation
                break

    for i, el in enumerate(res_representation):
        if el is not None:
            continue
        res_representation[i] = np.zeros(expected_shapes[i])
    return res_representation[1:]


def extract_per_level_tensor_representation_after_unshift(fillers_joined, max_tree_depth, role_shape, filler_shape):
    levels = []
    slicing_index = 0

    expected_shapes = generate_shapes(max_tree_depth=max_tree_depth,
                                      role_shape=role_shape,
                                      filler_shape=filler_shape)
    for shape in expected_shapes:
        max_index = np.prod(shape)
        product = fillers_joined[slicing_index:slicing_index + max_index].reshape(shape)
        slicing_index = slicing_index + max_index
        levels.append(product)
    return levels


def generate_shapes_for_unshift(max_tree_depth, role_shape, filler_shape):
    """
    ignoring the very first component that represents the root of the tree
    :param max_tree_depth:
    :param role_shape:
    :param filler_shape:
    :return:
    """
    return generate_shapes(max_tree_depth=max_tree_depth + 1,
                           role_shape=role_shape,
                           filler_shape=filler_shape)[1:]


if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()

    fillers_case_1 = np.array([
        [8, 0, 0],  # A
        [0, 15, 0],  # B
        [0, 0, 10],  # C
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

    MAX_TREE_DEPTH = 2
    SINGLE_ROLE_SHAPE = roles_case_1[0].shape
    SINGLE_FILLER_SHAPE = fillers_case_1[0].shape

    fillers_shapes = generate_shapes_for_unshift(max_tree_depth=MAX_TREE_DEPTH + 1,
                                                 role_shape=SINGLE_ROLE_SHAPE,
                                                 filler_shape=SINGLE_FILLER_SHAPE)

    original_structure = shifting_main()

    # TODO: hack works
    # original_structure[1][0][2][0][1] = 0.
    # original_structure[1][0][2][1][0] = 500.

    prepared_for_unshift = reshape_to_satisfy_max_depth_after_unshift(original_structure,
                                                                      MAX_TREE_DEPTH+1,
                                                                      SINGLE_ROLE_SHAPE,
                                                                      SINGLE_FILLER_SHAPE)

    keras_u0_unshifter = build_tree_unshifter_network(roles=dual_basic_roles_case_1, fillers_shapes=fillers_shapes)

    extracted_left_child = keras_u0_unshifter.predict_on_batch([
        *prepared_for_unshift
    ])

    print('calculated ex0()')

    extracted_left_child = extracted_left_child.reshape((*extracted_left_child.shape[1:],))
    left_child_tensor_representation = extract_per_level_tensor_representation_after_unshift(extracted_left_child,
                                                                                             max_tree_depth=MAX_TREE_DEPTH,
                                                                                             role_shape=SINGLE_ROLE_SHAPE,
                                                                                             filler_shape=SINGLE_FILLER_SHAPE)

    print('extracted A_x_r_0_C_x_r_1')
