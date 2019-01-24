import numpy as np
import keras.backend as K

from src.encoder.vendor.network import build_encoder_network as keras_encoder_network
from src.decoder.vendor.network import build_filler_decoder_network as keras_filler_decoder_network


def fill_until_square(not_square_matrix):
    cols = not_square_matrix.shape[1]
    rows = not_square_matrix.shape[0]
    new_rows = np.zeros((cols-rows, cols))
    res = np.vstack([not_square_matrix, new_rows])
    for i in range(res.shape[0]):
        res[i][i] = 1
    return res


if __name__ == '__main__':
    fillers = np.array([
        [110, 0, 0, 0],
        [0, 120, 0, 0],
        [0, 0, 130, 0],
    ])
    roles = np.array([
        [10, 0, 0, 0, 0],
        [0, 20, 0, 0, 0],
        [0, 0, 30, 0, 0],
    ])
    dim_fillers = fillers.shape[1]
    number_roles, dim_roles = roles.shape
    tensor_representation_shape = (dim_fillers, dim_roles)

    print('Building Keras encoder')
    fillers_shape = (*fillers.shape, 1)
    roles_shape = (*roles.shape, 1)

    keras_encoder = keras_encoder_network(input_shapes=(fillers_shape, roles_shape))

    print('Building Keras decoder')

    quadratized_roles = fill_until_square(roles)
    dual_roles = np.linalg.inv(quadratized_roles)

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
