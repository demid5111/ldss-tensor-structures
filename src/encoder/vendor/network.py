from keras.engine import Input
from keras.layers import Lambda, Permute, Add, Activation
from keras.models import Model


def mul_vec_on_vec(tensors):
    res = [tensors[0][i] * tensors[1][i] for i in range(tensors[0].shape[0])]
    return res


def build_encoder_network(input_shapes):
    transposer = Permute((2, 1))
    binding_cell = Lambda(mul_vec_on_vec)

    fillers_shape = input_shapes[0]
    roles_shape = input_shapes[1]

    input_fillers_layer = Input(shape=fillers_shape[1:], batch_shape=fillers_shape)
    input_roles_layer = Input(shape=roles_shape[1:], batch_shape=roles_shape)

    transposed_role_layer = transposer(input_roles_layer)
    binding_tensors_layer = binding_cell([input_fillers_layer, transposed_role_layer])

    summed_bindings = Add()(binding_tensors_layer)
    activation = Activation('relu')(summed_bindings)

    return Model(inputs=[input_fillers_layer, input_roles_layer], outputs=activation)
