import numpy as np

from core.peano.decode.vendor.network import build_decode_number_network
from core.peano.increment.vendor.network import build_increment_network
from core.peano.sum.vendor.network import build_sum_network
from core.peano.utils import number_to_tree, get_max_tree_depth
from core.utils import flattenize_per_tensor_representation
from demo.unshifting_structure import extract_per_level_tensor_representation_after_unshift


def decode_number(number_tree, fillers, dual_roles, max_depth):
    is_not_zero = True
    current_depth = max_depth
    current_number_tree = number_tree
    acc = 0
    while is_not_zero:
        keras_number_decoder = build_decode_number_network(fillers=fillers,
                                                           dual_roles=dual_roles,
                                                           max_depth=current_depth)

        flattened_number = flattenize_per_tensor_representation(current_number_tree)
        current_number_tree, is_not_zero_output = keras_number_decoder.predict_on_batch([
            *flattened_number
        ])

        acc += 1
        current_depth -= 1
        if is_not_zero_output[0][0] == 0:
            is_not_zero = False

    return acc


if __name__ == '__main__':
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
    MAX_TREE_DEPTH = get_max_tree_depth(MAX_NUMBER)
    SINGLE_ROLE_SHAPE = roles[0].shape
    SINGLE_FILLER_SHAPE = fillers[0].shape

    # TPR-Inc Network
    a = 0
    new_number_one = number_to_tree(a, MAX_TREE_DEPTH, fillers, roles)
    one_unshifted = flattenize_per_tensor_representation(new_number_one)

    keras_increment_network = build_increment_network(roles=roles,
                                                      dual_roles=dual_basic_roles_case_1,
                                                      fillers=fillers,
                                                      max_depth=MAX_TREE_DEPTH)
    print('Built increment network')

    new_number = keras_increment_network.predict_on_batch([
        one_unshifted
    ])

    new_number_tree = extract_per_level_tensor_representation_after_unshift(new_number, MAX_TREE_DEPTH,
                                                                            SINGLE_ROLE_SHAPE,
                                                                            SINGLE_FILLER_SHAPE)

    result_number = decode_number(number_tree=new_number_tree,
                                  fillers=fillers,
                                  dual_roles=dual_basic_roles_case_1,
                                  max_depth=MAX_TREE_DEPTH)
    print('After incrementing {}, get {}'.format(a, result_number))

    # TPR-Sum Network
    MAX_NUMBER = 4
    MAX_TREE_DEPTH = get_max_tree_depth(MAX_NUMBER)

    a = 1
    new_number_one = number_to_tree(a, MAX_TREE_DEPTH, fillers, roles)
    one_unshifted = flattenize_per_tensor_representation(new_number_one)

    b = 2
    new_number_two = number_to_tree(b, MAX_TREE_DEPTH, fillers, roles)
    two_unshifted = flattenize_per_tensor_representation(new_number_one)

    keras_sum_network = build_sum_network(roles=roles,
                                          dual_roles=dual_basic_roles_case_1,
                                          fillers=fillers,
                                          max_depth=MAX_TREE_DEPTH)
    print('Built increment network')

    new_number = keras_sum_network.predict_on_batch([
        one_unshifted,
        two_unshifted
    ])

    new_number_tree = extract_per_level_tensor_representation_after_unshift(new_number, MAX_TREE_DEPTH,
                                                                            SINGLE_ROLE_SHAPE,
                                                                            SINGLE_FILLER_SHAPE)

    result_number = decode_number(number_tree=new_number_tree,
                                  fillers=fillers,
                                  dual_roles=dual_basic_roles_case_1,
                                  max_depth=MAX_TREE_DEPTH)
    print('After {} + {}, get {}'.format(a, b, result_number))
