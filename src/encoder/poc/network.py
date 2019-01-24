from src.encoder.poc.ff_network import FeedForwardNetwork
from src.encoder.poc.layers.activation import Activation
from src.encoder.poc.layers.binding_cell import BindingCell
from src.encoder.poc.layers.eltwise import Eltwise
from src.encoder.poc.layers.input_data import InputData


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
