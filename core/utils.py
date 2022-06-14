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


def create_custom_constant(const_size, np_constant=None) -> np.ndarray:
    if np_constant is None:
        np_constant = np.zeros((const_size, 1))
    else:
        np_constant = np.reshape(np_constant, (*np_constant.shape, 1))
    return _reshape_to_batch(np_constant)


def create_matrix_constant(role, filler_len, max_depth, name, matrix_creator) -> np.ndarray:
    np_constant = matrix_creator(role, filler_len, max_depth, name)
    return _reshape_to_batch(np_constant)


def _reshape_to_batch(np_constant):
    batch_size = 1
    np_constant = np_constant.reshape((batch_size, *np_constant.shape))
    return np_constant
