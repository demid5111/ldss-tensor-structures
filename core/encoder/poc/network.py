import numpy as np

from core.encoder.poc.ff_network import FeedForwardNetwork
from core.encoder.poc.layers.activation import Activation
from core.encoder.poc.layers.binding_cell import BindingCell
from core.encoder.poc.layers.eltwise import Eltwise
from core.encoder.poc.layers.input_data import InputData


def build_encoder_network():
    b_cell = BindingCell()
    input_fillers = InputData()
    input_roles = InputData()
    elt_layer = Eltwise(t='sum')
    act_layer = Activation(t='ReLU')

    net = FeedForwardNetwork()
    net.add_input_layer(input_fillers)
    net.add_input_layer(input_roles)
    net.add_layer(b_cell, [input_fillers.id, input_roles.id])
    net.add_layer(elt_layer, [b_cell.id])
    net.add_layer(act_layer, [elt_layer.id], is_output=True)

    return net


if __name__ == '__main__':
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

    print('Running POC implementation')
    poc_net = build_encoder_network()
    poc_net.forward((fillers, roles))
    local_predictions = poc_net.outputs()[0]
