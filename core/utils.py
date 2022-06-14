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


def create_constant(role, filler_len, max_depth, name, matrix_creator):
    batch_size = 1
    np_constant = matrix_creator(role, filler_len, max_depth, name)
    np_constant = np_constant.reshape((batch_size, *np_constant.shape))
    return np_constant
