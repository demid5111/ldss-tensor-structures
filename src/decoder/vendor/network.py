import numpy as np

from keras.engine import Input
from keras.layers import Lambda, Permute
from keras.models import Model
import keras.backend as K


def mat_mul(tensors):
    return [K.dot(tensors[0][i],tensors[1][i]) for i in range(tensors[0].shape[0])]


def mat_transpose(matrix):
    return K.permute_dimensions(matrix, (1,0))


def build_filler_decoder_network(input_shapes):
    transposer = Permute((2, 1))
    transposer_matrix = Lambda(mat_transpose)
    unbinding_cell = Lambda(mat_mul)

    tensor_representation_shape = input_shapes[0]
    dual_roles_shape = input_shapes[1]

    input_tensor_representation_layer = Input(shape=tensor_representation_shape[1:],
                                              batch_shape=tensor_representation_shape)
    input_dual_roles_layer = Input(shape=dual_roles_shape[1:],
                                   batch_shape=dual_roles_shape)

    transposed_dual_role_layer = transposer(input_dual_roles_layer)
    binding_tensors_layer = unbinding_cell([
        input_tensor_representation_layer,
        transposed_dual_role_layer
    ])

    transposed_fillers = transposer_matrix(binding_tensors_layer)

    return Model(
        inputs=[
            input_tensor_representation_layer,
            input_dual_roles_layer
        ],
        outputs=transposed_fillers)
