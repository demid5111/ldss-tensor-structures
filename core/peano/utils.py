from core.active_passive_net.utils import prepare_input, elementary_join
from core.joiner.utils import generate_shapes
from core.joiner.vendor.network import build_tree_joiner_network


def get_max_tree_depth(maximum_number):
    """
    Returns the depth of the tree that is enough to represent maximum_number
    one is represented by one nesting level, therefore we add 1
    """
    return maximum_number + 1


def number_to_tree(target_number, max_tree_depth, fillers, roles):
    single_role_shape = roles[0].shape
    single_filler_shape = fillers[0].shape
    fillers_shapes = generate_shapes(max_tree_depth=max_tree_depth,
                                     role_shape=single_role_shape,
                                     filler_shape=single_filler_shape)

    if target_number == 0:
        return prepare_input(None, fillers_shapes)

    joiner_network = build_tree_joiner_network(roles=roles, fillers_shapes=fillers_shapes)

    one = fillers[0]
    for i in range(1):
        # one is a result of two joins
        # 1. join of filler and role_1 a.k.a zero representation
        # 2. join of step 1 and role_1
        one = elementary_join(joiner_network=joiner_network,
                              input_structure_max_shape=fillers_shapes,
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
                                 input_structure_max_shape=fillers_shapes,
                                 basic_roles=roles,
                                 basic_fillers=fillers,
                                 subtrees=(
                                     number,
                                     one
                                 ))
    return number
