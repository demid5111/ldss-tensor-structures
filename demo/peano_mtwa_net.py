import numpy as np

from core.joiner.vendor.network import build_tree_joiner_network
from core.peano.increment.vendor.network import build_increment_network, number_to_tree
from demo.active_passive_net import flattenize_per_tensor_representation
from demo.shifting_structure import generate_shapes
from demo.unshifting_structure import extract_per_level_tensor_representation_after_unshift

if __name__ == '__main__':
    print('need to get representation of the 1')

    # Input information
    fillers = np.array([
        [7, 0, 0, 0, 0],  # A
    ])

    roles = np.array([
        [10, 0],  # r_0
        [0, 5],  # r_1
    ])
    dual_basic_roles_case_1 = np.linalg.inv(roles)
    order_case_active = ['A', ]

    MAX_NUMBER = 4
    MAX_TREE_DEPTH = MAX_NUMBER + 2  # as one is already represented by structure with 2 levels
    SINGLE_ROLE_SHAPE = roles[0].shape
    SINGLE_FILLER_SHAPE = fillers[0].shape

    fillers_shapes = generate_shapes(max_tree_depth=MAX_TREE_DEPTH,
                                     role_shape=SINGLE_ROLE_SHAPE,
                                     filler_shape=SINGLE_FILLER_SHAPE)

    keras_joiner = build_tree_joiner_network(roles=roles, fillers_shapes=fillers_shapes)

    print('Potentially, this is representation of number greater than 1')
    TARGET_NUMBER = 2
    new_number_two = number_to_tree(TARGET_NUMBER, keras_joiner, fillers_shapes, fillers, roles, order_case_active)

    TARGET_NUMBER = 1
    new_number_one = number_to_tree(TARGET_NUMBER, keras_joiner, fillers_shapes, fillers, roles, order_case_active)
    one_unshifted = flattenize_per_tensor_representation(new_number_one)

    keras_increment_network = build_increment_network(roles=roles,
                                                      fillers=fillers,
                                                      dual_roles=dual_basic_roles_case_1,
                                                      max_depth=MAX_TREE_DEPTH,
                                                      increment_value=one_unshifted)
    print('Built increment network')

    two_unshifted = flattenize_per_tensor_representation(new_number_two)
    new_number_four = keras_increment_network.predict_on_batch([
        one_unshifted,
        one_unshifted
    ])

    print(new_number_four)
    four_tree = extract_per_level_tensor_representation_after_unshift(new_number_four, MAX_TREE_DEPTH,
                                                                      SINGLE_ROLE_SHAPE,
                                                                      SINGLE_FILLER_SHAPE)
    print(four_tree)

    TARGET_NUMBER = 4
    new_number_four = number_to_tree(TARGET_NUMBER, keras_joiner, fillers_shapes, fillers, roles, order_case_active)
    print(new_number_four)

    TARGET_NUMBER = 2
    new_number_two = number_to_tree(TARGET_NUMBER, keras_joiner, fillers_shapes, fillers, roles, order_case_active)
    print(new_number_two)
