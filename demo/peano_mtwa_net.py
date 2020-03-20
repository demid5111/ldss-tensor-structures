import numpy as np

from core.joiner.vendor.network import build_tree_joiner_network
from demo.active_passive_net import elementary_join, get_filler_by
from demo.shifting_structure import generate_shapes


def number_to_tree(target_number, joiner_network, maximum_shapes, fillers, roles, order_case_active):
    one = get_filler_by(name='A', order=order_case_active, fillers=fillers)
    for i in range(2):
        # one is a result of two joins
        # 1. join of filler and role a.k.a zero representation
        # 2. join of zero and a role a.k.a one representation
        one = elementary_join(joiner_network=joiner_network,
                              input_structure_max_shape=maximum_shapes,
                              basic_roles=roles,
                              basic_fillers=fillers,
                              subtrees=(
                                  one,
                                  None
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
    order_case_active = ['A', ]

    MAX_NUMBER = 4
    MAX_TREE_DEPTH = MAX_NUMBER + 2  # as one is already represented by structure with 2 levels
    SINGLE_ROLE_SHAPE = roles[0].shape
    SINGLE_FILLER_SHAPE = fillers[0].shape

    fillers_shapes = generate_shapes(max_tree_depth=MAX_TREE_DEPTH,
                                     role_shape=SINGLE_ROLE_SHAPE,
                                     filler_shape=SINGLE_FILLER_SHAPE)

    keras_joiner = build_tree_joiner_network(roles=roles, fillers_shapes=fillers_shapes)

    TARGET_NUMBER = 4
    print('Potentially, this is representation of number 4')
    print(number_to_tree(TARGET_NUMBER, keras_joiner, fillers_shapes, fillers, roles, order_case_active))
