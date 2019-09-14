"""
    We have two sentences:
    1. Few leaders are admired by George (Passive voice)
    2. George admires few leaders (Active voice)

    When using the notation from the original paper
    Legendre, G., Miyata, Y., & Smolensky, P. (1991).
    Distributed recursive structure processing.
    In Advances in Neural Information Processing Systems (pp. 591-597).
    Those trees are:

    Active voice
      (root)
    /         \
    A       /     \
          V       P

    Passive voice
      (root)
    /         \
    P       /           \
          /     \      /       \
        Aux     V     by        A


    We need to extract the following structure:

      (root)
    /         \
    V       /  \
          A     P
"""
import numpy as np

from core.joiner.vendor.network import build_tree_joiner_network
from demo.shifting_structure import generate_shapes, generate_input_placeholder
from demo.unshifting_structure import extract_per_level_tensor_representation


def prepare_input(subtree, max_shape):
    if subtree is None:
        # there is no subtree for this role, therefore just generate the placeholder
        return generate_input_placeholder(max_shape)
    subtree_shapes = np.array(tuple(np.array(i.shape) for i in subtree))

    if len(subtree_shapes) == len(max_shape) and \
            np.all([np.all(np.equal(subtree_shapes[i], max_shape[i])) for i, _ in enumerate(subtree_shapes)]):
        # TODO: need to understand why `extract_per_level_tensor_representation` returns a list
        # the subtree is already of a needed shape, just keep it unchanged
        return subtree

    if hasattr(subtree, 'shape') and len(subtree.shape) == 1:
        # TODO: need to understand why `extract_per_level_tensor_representation` returns a list
        # subtree is a simple filler
        placeholder = generate_input_placeholder(max_shape)
        placeholder[0] = subtree.reshape(1, *subtree.shape)
        return placeholder

    raise NotImplementedError('This subtree cannot be prepared for join')


def elementary_join(input_structure_max_shape, roles, subtrees):
    input_tensors = map(lambda s: prepare_input(s, input_structure_max_shape), subtrees)

    fillers_joined = keras_joiner.predict_on_batch([i for p in input_tensors for i in p])

    single_role_shape = roles[0].shape
    single_filler_shape = list(filter(lambda el: el is not None, subtrees))[0].shape
    max_depth = input_structure_max_shape.shape[0]
    # TODO: maximum shape should be found from the max shape of input structure
    return extract_per_level_tensor_representation(fillers_joined,
                                                   max_tree_depth=max_depth,
                                                   role_shape=single_role_shape,
                                                   filler_shape=single_filler_shape)


def get_filler_by(name, order, fillers):
    return fillers[order.index(name)]


if __name__ == '__main__':
    print('Hello, Active-Passive Net')

    # Input information
    fillers = np.array([
        [7, 0, 0, 0, 0],  # A
        [0, 4, 0, 0, 0],  # V
        [0, 0, 2, 0, 0],  # P
        [0, 0, 0, 5, 0],  # Aux
        [0, 0, 0, 0, 3],  # by
    ])
    roles = np.array([
        [10, 0],  # r_0
        [0, 5],  # r_1
    ])
    mapping_case_active = {
        'A': [0],
        'V': [0, 1],
        'P': [1, 1]
    }
    order_case_active = ['A', 'V', 'P']

    mapping_case_passive = {
        'P': [0],
        'Aux': [0, 0, 1],
        'V': [1, 0, 1],
        'by': [0, 1, 1],
        'A': [1, 1, 1],
    }
    order_case_passive = ['A', 'V', 'P', 'Aux', 'by']

    MAX_TREE_DEPTH = 3
    SINGLE_ROLE_SHAPE = roles[0].shape
    SINGLE_FILLER_SHAPE = fillers[0].shape

    fillers_shapes = generate_shapes(max_tree_depth=MAX_TREE_DEPTH,
                                     role_shape=SINGLE_ROLE_SHAPE,
                                     filler_shape=SINGLE_FILLER_SHAPE)

    keras_joiner = build_tree_joiner_network(roles=roles, fillers_shapes=fillers_shapes)

    t_V_x_r_0_P_x_r_1 = elementary_join(input_structure_max_shape=fillers_shapes,
                                        roles=roles,
                                        subtrees=(
                                            get_filler_by(name='V', order=order_case_active, fillers=fillers),
                                            get_filler_by(name='P', order=order_case_active, fillers=fillers)
                                        ))
    print('calculated cons(V,P)')

    t_A_r0_V_r0r1_P_r1r1 = elementary_join(input_structure_max_shape=fillers_shapes,
                                           roles=roles,
                                           subtrees=(
                                               get_filler_by(name='A', order=order_case_active, fillers=fillers),
                                               t_V_x_r_0_P_x_r_1
                                           ))
    print('calculated cons(A,cons(V,P))')
    print('Found tensor representation of the Active Voice sentence')
