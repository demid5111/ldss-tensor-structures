import numpy as np
import keras.backend as K

from src.encoder.poc.network import build_encoder_network as poc_encoder_network
from src.encoder.vendor.network import build_encoder_network as keras_encoder_network


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
    poc_net = poc_encoder_network()
    poc_net.forward((fillers, roles))
    local_predictions = poc_net.outputs()[0]

    print('Running Keras implementation')
    fillers_shape = (*fillers.shape, 1)
    roles_shape = (*roles.shape, 1)

    reshaped_fillers = fillers.reshape(fillers_shape)
    reshaped_roles = roles.reshape(roles_shape)

    keras_net = keras_encoder_network(input_shapes=(fillers_shape, roles_shape))
    with K.get_session():
        keras_predictions = keras_net.predict_on_batch([reshaped_fillers, reshaped_roles])

    report = '' if np.allclose(local_predictions, keras_predictions) else 'not '

    print('Comparing own and Keras models: {}identical'.format(report))
