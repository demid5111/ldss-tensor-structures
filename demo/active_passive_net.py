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
from demo.shifting_structure import generate_shapes, generate_input_placeholder, extract_per_level_tensor_representation


def shift_by_role(f_shapes, max_depth, role_index, role, filler):
    single_filler_shape = filler.shape
    single_role_shape = role.shape
    placeholders = (
        generate_input_placeholder(f_shapes),
        generate_input_placeholder(f_shapes)
    )
    placeholders[role_index][0] = filler.reshape(1, *single_filler_shape)

    inputs = []
    for p in placeholders:
        for i in p:
            inputs.append(i)

    fillers_joined = keras_joiner.predict_on_batch(inputs)

    return extract_per_level_tensor_representation(fillers_joined,
                                                   max_tree_depth=max_depth,
                                                   role_shape=single_role_shape,
                                                   filler_shape=single_filler_shape)


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

    t_A_x_r_0 = shift_by_role(f_shapes=fillers_shapes,
                              max_depth=MAX_TREE_DEPTH,
                              role_index=0,
                              role=roles[0],
                              filler=fillers[0])

    print(t_A_x_r_0)
    print('calculated cons (A _x_ r_0)')
