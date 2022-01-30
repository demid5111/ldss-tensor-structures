import numpy as np
import tensorflow as tf

def flattenize_per_tensor_representation(blobs):
    length = sum(map(lambda b: b.size, blobs))
    flattened_res = np.zeros((length,))
    last_index = 0
    for blob in blobs:
        flattened_res[last_index: last_index + blob.size] = blob.flatten()
        last_index = last_index + blob.size
    return flattened_res


def get_filler_by(name, order, fillers):
    return fillers[order.index(name)]


def keras_constant_layer(np_constant, name, batch_size=1):
    np_constant = np_constant.reshape((batch_size, *np_constant.shape))
    tf_constant = tf.keras.backend.constant(np_constant, dtype='float32')
    return tf.keras.layers.Input(tensor=tf_constant, shape=np_constant.shape, dtype='float32', name=name)
