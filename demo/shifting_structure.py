import numpy as np

from core.joiner.utils import generate_shapes, generate_input_placeholder, \
    extract_per_level_tensor_representation_after_shift, reshape_to_satisfy_max_depth_after_shift
from core.joiner.vendor.network import build_tree_joiner_network


def sum_tensors(left, right):
    return [l + r for l, r in zip(left, right)]


def main():
    """
        First use case for the structure that should be shifted left

        Starting from:
        root

        We want to get:
        root
        |
        A (left-child-of-root)
        """
    fillers_case_1 = np.array([
        [8, 0, 0],  # A
        [0, 15, 0],  # B
        [0, 0, 10],  # C
    ])
    roles_case_1 = np.array([
        [10, 0],  # r_0
        [0, 5],  # r_1
    ])

    MAX_TREE_DEPTH = 2
    SINGLE_ROLE_SHAPE = roles_case_1[0].shape
    SINGLE_FILLER_SHAPE = fillers_case_1[0].shape

    fillers_shapes = generate_shapes(max_tree_depth=MAX_TREE_DEPTH,
                                     role_shape=SINGLE_ROLE_SHAPE,
                                     filler_shape=SINGLE_FILLER_SHAPE)

    keras_joiner = build_tree_joiner_network(roles=roles_case_1, fillers_shapes=fillers_shapes)

    left_subtree_placeholder = generate_input_placeholder(fillers_shapes)
    left_subtree_placeholder[0] = fillers_case_1[0].reshape(1, *SINGLE_FILLER_SHAPE)
    right_subtree_placeholder = generate_input_placeholder(fillers_shapes)

    fillers_joined = keras_joiner.predict_on_batch([
        *left_subtree_placeholder,
        *right_subtree_placeholder
    ])
    print('calculated cons (A _x_ r_0)')

    """
    Second use case for the structure that should be shifted left
    (continued from the first case)

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
    tensor_repr_A_x_r_0 = extract_per_level_tensor_representation_after_shift(fillers_joined,
                                                                              max_tree_depth=MAX_TREE_DEPTH,
                                                                              role_shape=SINGLE_ROLE_SHAPE,
                                                                              filler_shape=SINGLE_FILLER_SHAPE)
    print('split per layer')

    prepared_for_shift = reshape_to_satisfy_max_depth_after_shift(tensor_repr_A_x_r_0,
                                                                  MAX_TREE_DEPTH,
                                                                  SINGLE_ROLE_SHAPE,
                                                                  SINGLE_FILLER_SHAPE)
    print('reshaped for second shift')

    fillers_joined_second_case = keras_joiner.predict_on_batch([
        *prepared_for_shift,
        *right_subtree_placeholder
    ])
    print('calculated cons (A _x_ r_0 _x_ r_0)')

    tensor_repr_A_x_r_0_x_r_0 = extract_per_level_tensor_representation_after_shift(fillers_joined_second_case,
                                                                                    max_tree_depth=MAX_TREE_DEPTH,
                                                                                    role_shape=SINGLE_ROLE_SHAPE,
                                                                                    filler_shape=SINGLE_FILLER_SHAPE)
    print('split per layer after second case')

    """
    Third use case for the structure that should be shifted left
    (continued from the second case)

    Starting from:
    root
    |
    A (left-child-of-left-child-of-root)

    We want to get:
    root
    |   \
    |   B (right-child-of-root)
    A (left-child-of-left-child-of-root)    
    """

    prepared_for_shift_A_x_r_0 = reshape_to_satisfy_max_depth_after_shift(tensor_repr_A_x_r_0,
                                                                          MAX_TREE_DEPTH,
                                                                          SINGLE_ROLE_SHAPE,
                                                                          SINGLE_FILLER_SHAPE)

    right_subtree_placeholder = generate_input_placeholder(fillers_shapes)
    right_subtree_placeholder[0] = fillers_case_1[1].reshape(1, *SINGLE_FILLER_SHAPE)

    fillers_joined_third_case = keras_joiner.predict_on_batch([
        *prepared_for_shift_A_x_r_0,
        *right_subtree_placeholder
    ])

    tensor_repr_A_x_r_0_x_r_0_B_x_r_1 = extract_per_level_tensor_representation_after_shift(fillers_joined_third_case,
                                                                                            max_tree_depth=MAX_TREE_DEPTH,
                                                                                            role_shape=SINGLE_ROLE_SHAPE,
                                                                                            filler_shape=SINGLE_FILLER_SHAPE)
    print('split per layer after third case')

    """
    Fourth use case for the structure that should be shifted left
    (continued from the third case)

    Starting from:
    root
    |   \
    |   B (right-child-of-root)
    A (left-child-of-left-child-of-root) 

    We want to get:
    root
    |                                                               \
    |                                       \                       B (right-child-of-root)
    A (left-child-of-left-child-of-root)    C (right-child-of-left-child-of-root)
    """

    left_subtree_placeholder = generate_input_placeholder(fillers_shapes)
    right_subtree_placeholder = generate_input_placeholder(fillers_shapes)
    right_subtree_placeholder[0] = fillers_case_1[2].reshape(1, *SINGLE_FILLER_SHAPE)

    fillers_joined_fourth_case_simple_c = keras_joiner.predict_on_batch([
        *left_subtree_placeholder,
        *right_subtree_placeholder
    ])

    tensor_repr_C_x_r_1 = extract_per_level_tensor_representation_after_shift(fillers_joined_fourth_case_simple_c,
                                                                              max_tree_depth=MAX_TREE_DEPTH,
                                                                              role_shape=SINGLE_ROLE_SHAPE,
                                                                              filler_shape=SINGLE_FILLER_SHAPE)

    right_subtree_placeholder = generate_input_placeholder(fillers_shapes)

    prepared_for_shift_C_x_r_1 = reshape_to_satisfy_max_depth_after_shift(tensor_repr_C_x_r_1,
                                                                          MAX_TREE_DEPTH,
                                                                          SINGLE_ROLE_SHAPE,
                                                                          SINGLE_FILLER_SHAPE)

    fillers_joined_fourth_case_complex_c = keras_joiner.predict_on_batch([
        *prepared_for_shift_C_x_r_1,
        *right_subtree_placeholder,
    ])

    tensor_repr_C_x_r_1_x_r_0 = extract_per_level_tensor_representation_after_shift(
        fillers_joined_fourth_case_complex_c,
        max_tree_depth=MAX_TREE_DEPTH,
        role_shape=SINGLE_ROLE_SHAPE,
        filler_shape=SINGLE_FILLER_SHAPE)

    tree_representation = sum_tensors(tensor_repr_C_x_r_1_x_r_0, tensor_repr_A_x_r_0_x_r_0_B_x_r_1)
    print('calculated tree representation')

    prepared_for_shift_tree_representation = reshape_to_satisfy_max_depth_after_shift(tree_representation,
                                                                                      MAX_TREE_DEPTH+1,
                                                                                      SINGLE_ROLE_SHAPE,
                                                                                      SINGLE_FILLER_SHAPE)
    return prepared_for_shift_tree_representation


if __name__ == '__main__':
    main()
