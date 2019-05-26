import numpy as np

from recursive_structure import pre_process_roles
from src.joiner.vendor.network import build_tree_joiner_network


def generate_shapes(max_tree_depth, role_shape, filler_shape):
    shapes = []
    for i in range(max_tree_depth):
        roles_shape_addon = [role_shape[0] for _ in range(i)]
        shapes.append(np.array([1, filler_shape[0], *roles_shape_addon]))
    return np.array(shapes)


def generate_input_placeholder(fillers_shapes):
    return [np.zeros(shape) for shape in fillers_shapes]


def extract_per_level_tensor_representation(fillers_joined, max_tree_depth, role_shape, filler_shape):
    levels = []
    slicing_index = 0

    expected_shapes = generate_shapes(max_tree_depth=max_tree_depth + 1,
                                      role_shape=role_shape,
                                      filler_shape=filler_shape)
    for shape in expected_shapes[1:]:
        max_index = np.prod(shape)
        product = fillers_joined[slicing_index:slicing_index + max_index].reshape(shape)
        slicing_index = slicing_index + max_index
        levels.append(product)
    return levels


def reshape_to_satisfy_max_depth(tensor_representation, max_tree_depth, role_shape, filler_shape):
    expected_shapes = generate_shapes(max_tree_depth=max_tree_depth,
                                      role_shape=role_shape,
                                      filler_shape=filler_shape)

    res_representation = [None for _ in range(max_tree_depth)]
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
    return res_representation


def sum_tensors(left, right):
    return [l+r for l,r in zip(left, right)]


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
    expected_shapes = generate_shapes(max_tree_depth=MAX_TREE_DEPTH,
                                      role_shape=SINGLE_ROLE_SHAPE,
                                      filler_shape=SINGLE_FILLER_SHAPE)
    tensor_repr_A_x_r_0 = extract_per_level_tensor_representation(fillers_joined,
                                                                  max_tree_depth=MAX_TREE_DEPTH,
                                                                  role_shape=SINGLE_ROLE_SHAPE,
                                                                  filler_shape=SINGLE_FILLER_SHAPE)
    print('split per layer')

    prepared_for_shift = reshape_to_satisfy_max_depth(tensor_repr_A_x_r_0,
                                                      MAX_TREE_DEPTH,
                                                      SINGLE_ROLE_SHAPE,
                                                      SINGLE_FILLER_SHAPE)
    print('reshaped for second shift')

    fillers_joined_second_case = keras_joiner.predict_on_batch([
        *prepared_for_shift,
        *right_subtree_placeholder
    ])
    print('calculated cons (A _x_ r_0 _x_ r_0)')

    tensor_repr_A_x_r_0_x_r_0 = extract_per_level_tensor_representation(fillers_joined_second_case,
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

    prepared_for_shift_A_x_r_0 = reshape_to_satisfy_max_depth(tensor_repr_A_x_r_0,
                                                              MAX_TREE_DEPTH,
                                                              SINGLE_ROLE_SHAPE,
                                                              SINGLE_FILLER_SHAPE)

    right_subtree_placeholder = generate_input_placeholder(fillers_shapes)
    right_subtree_placeholder[0] = fillers_case_1[1].reshape(1, *SINGLE_FILLER_SHAPE)

    fillers_joined_third_case = keras_joiner.predict_on_batch([
        *prepared_for_shift_A_x_r_0,
        *right_subtree_placeholder
    ])

    tensor_repr_A_x_r_0_x_r_0_B_x_r_1 = extract_per_level_tensor_representation(fillers_joined_third_case,
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

    tensor_repr_C_x_r_1 = extract_per_level_tensor_representation(fillers_joined_fourth_case_simple_c,
                                                                  max_tree_depth=MAX_TREE_DEPTH,
                                                                  role_shape=SINGLE_ROLE_SHAPE,
                                                                  filler_shape=SINGLE_FILLER_SHAPE)

    right_subtree_placeholder = generate_input_placeholder(fillers_shapes)

    prepared_for_shift_C_x_r_1 = reshape_to_satisfy_max_depth(tensor_repr_C_x_r_1,
                                                              MAX_TREE_DEPTH,
                                                              SINGLE_ROLE_SHAPE,
                                                              SINGLE_FILLER_SHAPE)

    fillers_joined_fourth_case_complex_c = keras_joiner.predict_on_batch([
        *prepared_for_shift_C_x_r_1,
        *right_subtree_placeholder,
    ])

    tensor_repr_C_x_r_1_x_r_0 = extract_per_level_tensor_representation(fillers_joined_fourth_case_complex_c,
                                                                        max_tree_depth=MAX_TREE_DEPTH,
                                                                        role_shape=SINGLE_ROLE_SHAPE,
                                                                        filler_shape=SINGLE_FILLER_SHAPE)

    tree_representation = sum_tensors(tensor_repr_C_x_r_1_x_r_0, tensor_repr_A_x_r_0_x_r_0_B_x_r_1)
    print('calculated tree representation')
    return tree_representation


if __name__ == '__main__':
    main()
