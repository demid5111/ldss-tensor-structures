import numpy as np

from core.joiner.vendor.network import build_tree_joiner_network
from core.peano.increment.vendor.network import build_increment_network
from demo.active_passive_net import flattenize_per_tensor_representation, prepare_input, get_filler_by, elementary_join
from demo.shifting_structure import generate_shapes
from demo.unshifting_structure import extract_per_level_tensor_representation_after_unshift


def number_to_tree(target_number, joiner_network, maximum_shapes, fillers, roles, order_case_active):
    if target_number == 0:
        return prepare_input(None, maximum_shapes)

    one = get_filler_by(name='A', order=order_case_active, fillers=fillers)
    for i in range(1):
        # one is a result of two joins
        # 1. join of filler and role_1 a.k.a zero representation
        # 2. join of step 1 and role_1 a.k.a zero representation
        one = elementary_join(joiner_network=joiner_network,
                              input_structure_max_shape=maximum_shapes,
                              basic_roles=roles,
                              basic_fillers=fillers,
                              subtrees=(
                                  None,
                                  one
                              ))

    number = one
    for i in range(target_number - 1):
        # easy as 2 is just one join of (one+one)
        # easy as 3 is just two joins: (one+one)+one
        number = elementary_join(joiner_network=joiner_network,
                                 input_structure_max_shape=maximum_shapes,
                                 basic_roles=roles,
                                 basic_fillers=fillers,
                                 subtrees=(
                                     number,
                                     one
                                 ))
    return number


def get_max_tree_depth(maximum_number):
    """
    Returns the depth of the tree that is enough to represent maximum_number
    one is represented by one nesting level, therefore we add 1
    """
    return maximum_number + 1


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

    MAX_NUMBER = 2
    MAX_TREE_DEPTH = get_max_tree_depth(MAX_NUMBER)  # as one is already represented by structure with 2 levels
    SINGLE_ROLE_SHAPE = roles[0].shape
    SINGLE_FILLER_SHAPE = fillers[0].shape

    fillers_shapes = generate_shapes(max_tree_depth=MAX_TREE_DEPTH,
                                     role_shape=SINGLE_ROLE_SHAPE,
                                     filler_shape=SINGLE_FILLER_SHAPE)

    keras_joiner = build_tree_joiner_network(roles=roles, fillers_shapes=fillers_shapes)

    TARGET_NUMBER = 1
    new_number_one = number_to_tree(TARGET_NUMBER, keras_joiner, fillers_shapes, fillers, roles, order_case_active)
    one_unshifted = flattenize_per_tensor_representation(new_number_one)

    keras_increment_network = build_increment_network(roles=roles,
                                                      fillers=fillers,
                                                      dual_roles=dual_basic_roles_case_1,
                                                      max_depth=MAX_TREE_DEPTH,
                                                      increment_value=one_unshifted)
    print('Built increment network')

    new_number = keras_increment_network.predict_on_batch([
        one_unshifted,
        one_unshifted
    ])

    print(new_number)
    four_tree = extract_per_level_tensor_representation_after_unshift(new_number, MAX_TREE_DEPTH,
                                                                      SINGLE_ROLE_SHAPE,
                                                                      SINGLE_FILLER_SHAPE)
    print(four_tree)

    TARGET_NUMBER = 0
    new_number_two = number_to_tree(TARGET_NUMBER, keras_joiner, fillers_shapes, fillers, roles, order_case_active)
    print(new_number_two)

    TARGET_NUMBER = 2
    new_number_two = number_to_tree(TARGET_NUMBER, keras_joiner, fillers_shapes, fillers, roles, order_case_active)
    print(new_number_two)
