import numpy as np
from keras import optimizers
from keras.engine import Input
from keras.layers import Lambda, Permute, Add, Activation
from keras.models import Model
import keras.backend as K

from math_utils import feed_forward_recursion_matrix, \
    single_dimension_transfer_weights
from src.ff_network import FeedForwardNetwork
from src.layers.activation import Activation as ActivationLayer
from src.layers.binding_cell import BindingCell
from src.layers.eltwise import Eltwise
from src.layers.input_data import InputData


def mul_vec_on_vec(tensors):
    res = [tensors[0][i] * tensors[1][i] for i in range(len(fillers))]
    return res


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

    print('Running the local implementation')
    net = FeedForwardNetwork()
    fillers = np.array([
        [1, 0, 0, 0],
        [0.5, 3, 0, 0],
        [0.6, 0, 2, 0],
    ])
    roles = np.array([
        [0, 0, 0, 0.1, 0],
        [0, 0.3, 0, 0, 0],
        [0, 0, 0.9, 0, 0],
    ])

    b_cell = BindingCell()
    input_fillers = InputData()
    input_roles = InputData()
    elt_layer = Eltwise(t='sum')
    act_layer = ActivationLayer(t='ReLU')

    net.add_input_layer(input_fillers)
    net.add_input_layer(input_roles)
    net.add_layer(b_cell, [input_fillers.id, input_roles.id])
    net.add_layer(elt_layer, [b_cell.id])
    net.add_layer(act_layer, [elt_layer.id], is_output=True)

    net.dump_structure()

    net.fill_input(input_fillers.id, fillers)
    net.fill_input(input_roles.id, roles)

    net.forward()
    local_predictions = net.outputs()[0]

    print('Running the keras implementation')
    transposer = Permute((2,1))
    binding_cell = Lambda(mul_vec_on_vec)

    fillers_shape = (*fillers.shape, 1)
    roles_shape = (*roles.shape, 1)

    input_fillers_layer = Input(shape=fillers_shape[1:], batch_shape=fillers_shape)
    input_roles_layer = Input(shape=roles_shape[1:], batch_shape=roles_shape)

    transposed_role_layer = transposer(input_roles_layer)
    binding_tensors_layer = binding_cell([input_fillers_layer, transposed_role_layer])

    summed_bindings = Add()(binding_tensors_layer)
    activation = Activation('relu')(summed_bindings)

    y = Model(inputs=[input_fillers_layer, input_roles_layer], outputs=activation)

    reshaped_fillers = fillers.reshape(fillers_shape)
    reshaped_roles = roles.reshape(roles_shape)
    keras_predictions = y.predict_on_batch([reshaped_fillers, reshaped_roles])

    report = '' if np.allclose(local_predictions, keras_predictions) else 'not '
    print('Comparing Local and Keras: {}identical'.format(report))
