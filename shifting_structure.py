import numpy as np

from recursive_structure import pre_process_roles
from src.joiner.vendor.network import build_tree_joiner_network


def generate_shapes(max_tree_depth, role_shape, filler_shape):
    shapes = []
    for i in range(max_tree_depth):
        roles_shape_addon = [role_shape[0] for j in range(i)]
        shapes.append(np.array([1, filler_shape[0], *roles_shape_addon]))
    return np.array(shapes)


def generate_input_placeholder(fillers_shapes):
    return [np.zeros(shape) for shape in fillers_shapes]


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

    keras_joiner = build_tree_joiner_network(roles=roles_case_1, fillers_shapes=fillers_shapes)

    left_subtree_placeholder = generate_input_placeholder(fillers_shapes)
    left_subtree_placeholder[0] = fillers_case_1[0].reshape(1, *fillers_case_1[0].shape)
    right_subtree_placeholder = generate_input_placeholder(fillers_shapes)

    fillers_joined = keras_joiner.predict_on_batch([
        *left_subtree_placeholder,
        *right_subtree_placeholder
    ])
    print('calculated cons')

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
