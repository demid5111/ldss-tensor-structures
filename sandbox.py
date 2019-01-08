import numpy as np

from math_utils import feed_forward_recursion_matrix, \
    single_dimension_transfer_weights
from src.ff_network import FeedForwardNetwork
from src.layers.binding_cell import BindingCell

if __name__ == '__main__':
    # roles_basis = np.array([
    #     [10, -3],  # r_{0}
    #     [3, 10]  # r_{1}
    # ])
    #
    # assert np.sum(np.prod(roles_basis, axis=0)) == 0, 'Roles vectors should be orthogonal'
    #
    # filler_basis = np.array([
    #     [2, 0, 1],  # A
    #     [4, 10, 0],  # B
    #     [0, 3, 5]  # C
    # ])
    #
    # assert np.sum(np.prod(filler_basis, axis=0)) == 0, 'Filler vectors should be orthogonal'
    #
    # # A (x) r_{0} + [B (x) r_{0} + C (x) r_{1}] (x) r_{1}
    #
    # # encoding the structure, 3 is the maximum depth
    # #               epsilon
    # #           /                   \
    # #       A                   /       \
    # #                          B        C
    # MAXIMUM_TREE_DEPTH = 3
    # W = np.zeros((MAXIMUM_TREE_DEPTH, MAXIMUM_TREE_DEPTH))
    #
    # # 1. Make A (x) r_0
    #
    # # 1.1. Prepare weights
    #
    # # 1.1.1 Prepare translation from level 2 to level 3
    # res = single_dimension_transfer_weights(level=1,
    #                                         filler_dim=filler_basis[0].shape[0],
    #                                         role_v=roles_basis[0])
    #
    # feed_forward_recursion_matrix(MAXIMUM_TREE_DEPTH, roles_basis[0].shape[0])

    net = FeedForwardNetwork()

    b_cell = BindingCell()
    elt_layer = SomeEltwise()
    act_layer = SomeActivation()

    net.add_input_layer(b_cell)
    net.add_layer(elt_layer, [b_cell])
    net.add_layer(act_layer, [elt_layer], is_output=True)

    net.add_input(fillers)
    net.add_input(roles)

    net.forward()
    print(net.outputs()[0])
