import numpy as np


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
