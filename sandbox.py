import numpy as np
import keras.backend as K

from src.encoder.vendor.network import build_encoder_network as keras_encoder_network
from src.decoder.vendor.network import build_filler_decoder_network as keras_filler_decoder_network


if __name__ == '__main__':
    fillers = np.array([
        [8, 0, 0],
        [0, 15, 0],
        [0, 0, 10],
    ])
    roles = np.array([
        [10, 0, 0],
        [0, 5, 0],
        [0, 0, 15],
    ])
    number_fillers, dim_fillers = fillers.shape
    number_roles, dim_roles = roles.shape

    assert number_fillers == dim_fillers, 'Fillers should be a quadratic matrix'
    assert number_roles == dim_fillers, 'Roles should be a quadratic matrix'

    tensor_representation_shape = (dim_fillers, dim_roles)

    print('Building Keras encoder')
    fillers_shape = (*fillers.shape, 1)
    roles_shape = (*roles.shape, 1)

    keras_encoder = keras_encoder_network(input_shapes=(fillers_shape, roles_shape))

    print('Building Keras decoder')

    dual_roles = np.linalg.inv(roles)

    tensor_representation_3d_shape = (1, *tensor_representation_shape)
    dual_roles_3d_shape = (1, *roles.shape)

    input_shapes = (tensor_representation_3d_shape, dual_roles_3d_shape)
    keras_decoder = keras_filler_decoder_network(input_shapes)

    with K.get_session():
        print('Running Keras encoder')

        reshaped_fillers = fillers.reshape(fillers_shape)
        reshaped_roles = roles.reshape(roles_shape)

        tensor_representation = keras_encoder.predict_on_batch([
            reshaped_fillers,
            reshaped_roles
        ])

        print('Structural tensor representation')
        print(tensor_representation)

        print('Running Keras decoder')

        reshaped_tensor_representation = tensor_representation.reshape(tensor_representation_3d_shape)
        reshaped_dual_roles = dual_roles[:number_roles].reshape(dual_roles_3d_shape)

        fillers_restored = keras_decoder.predict_on_batch([
            reshaped_tensor_representation,
            reshaped_dual_roles
        ])

    for i, filler in enumerate(fillers_restored):
        print('Original: '
              '\n\t[role]({}) with \t\t\t\t[filler]({}). '
              '\n\tDecoded: with [dual role]({})\t\t\t[filler]({})'
              .format(roles[i], fillers[i], dual_roles[i], filler))
