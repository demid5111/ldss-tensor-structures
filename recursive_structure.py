"""
Use case for the structure that has variable nesting

Using this structure for demonstration:
root
|  \
A  / \
  B   \
     / \
    C  D
"""
from functools import reduce

import numpy as np
import keras.backend as K

from src.encoder.vendor.network import build_encoder_network as keras_encoder_network
from src.decoder.vendor.network import build_filler_decoder_network as keras_filler_decoder_network


def calculate_tensor_representation_shape(fillers_vector_size, roles_vector_size, max_depth):
    return 1, fillers_vector_size * roles_vector_size ** max_depth


def calculate_tensor_representation_shape_as_2d_matrix(fillers_vector_size, roles_vector_size, max_depth):
    return fillers_vector_size, roles_vector_size ** max_depth


def find_maximum_depth(descr):
    return len(sorted(descr.values(), key=lambda x: len(x), reverse=True)[0])


def accumulate_roles(roles_vectors, filler_roles_mapping):
    roles = {}
    for filler_name, role_hierarchy in filler_roles_mapping.items():
        if len(role_hierarchy) == 1:
            roles[filler_name] = roles_vectors[role_hierarchy[0]]
        else:
            roles[filler_name] = reduce(
                lambda x, y:
                np.tensordot(roles_vectors[x], roles_vectors[y], axes=0)
                if isinstance(x, int)
                else np.tensordot(x, roles_vectors[y], axes=0),
                role_hierarchy)
    return roles


def flatten_roles(accumulated_roles):
    return {filler_name: accumulated_role.flatten() for filler_name, accumulated_role in accumulated_roles.items()}


def equalize_roles_length(flattened_roles):
    max_size = len(max(flattened_roles.values(), key=lambda x: len(x)))
    roles = {}
    for filler_name, flattened_role in flattened_roles.items():
        tmp = np.copy(flattened_role)
        tmp.resize((max_size,), refcheck=False)
        roles[filler_name] = tmp
    return roles


def fill_to_square(final_processed_roles):
    # tmp = np.copy(final_processed_roles)
    # tmp.resize((tmp.shape[1], tmp.shape[1]), refcheck=False)
    # tmp[7][7] = 1
    new_shape = final_processed_roles.shape[1] - final_processed_roles.shape[0], final_processed_roles.shape[1]
    # addon = np.random.rand(*new_shape)
    # addon = np.random.randn(*new_shape)
    # addon /= 100000

    addon = np.zeros(new_shape)
    for row_index in range(new_shape[0]):
        for column_index in range(new_shape[1]):
            if column_index != (row_index + final_processed_roles.shape[0]) and row_index != (column_index - final_processed_roles.shape[0]):
                continue
            addon[row_index][column_index] = 0.000001
    tmp = np.concatenate((np.copy(final_processed_roles), addon), axis=0)
    return tmp


def filter_dual_roles(dual_roles):
    return np.array([i for i in dual_roles if not np.all(i == 0)])


if __name__ == '__main__':
    # Input information
    FILLERS_VECTORS = np.array([
        [8, 0, 0, 0],  # A
        [0, 15, 0, 0],  # B
        [0, 0, 10, 0],  # C
        [0, 0, 0, 3],  # D
    ])
    ROLES_VECTORS = np.array([
        [10, 0],  # r_0
        [0, 5],  # r_1
    ])
    FILLER_ROLES_MAPPING = {
        'A': [0],
        'B': [0, 1],
        'C': [0, 1, 1],
        'D': [1, 1, 1]
    }
    ORDER = ['A', 'B', 'C', 'D']

    # Processing logic
    maximum_depth = find_maximum_depth(FILLER_ROLES_MAPPING)
    number_fillers, dim_fillers = FILLERS_VECTORS.shape
    number_basic_roles, dim_roles = ROLES_VECTORS.shape

    number_roles = len(list(FILLER_ROLES_MAPPING.values()))

    assert number_fillers == dim_fillers, 'Fillers should be a quadratic matrix'
    assert number_basic_roles == dim_roles, 'Roles should be a quadratic matrix'

    tensor_representation_shape = calculate_tensor_representation_shape_as_2d_matrix(dim_fillers, dim_roles, maximum_depth)

    accumulated_roles = accumulate_roles(ROLES_VECTORS, FILLER_ROLES_MAPPING)
    flattened_roles = flatten_roles(accumulated_roles)
    equal_length_roles = equalize_roles_length(flattened_roles)

    final_processed_roles = np.array([
        equal_length_roles['A'],
        equal_length_roles['B'],
        equal_length_roles['C'],
        equal_length_roles['D']
    ])

    print('Building Keras encoder')
    fillers_shape = (*FILLERS_VECTORS.shape, 1)
    # here we should already have some shape like 4, 8 ,1
    roles_shape = (*final_processed_roles.shape, 1)

    keras_encoder = keras_encoder_network(input_shapes=(fillers_shape, roles_shape))

    print('Building Keras decoder')

    tensor_representation_3d_shape = (1, *tensor_representation_shape)

    square_roles_matrix = fill_to_square(final_processed_roles)
    dual_roles = np.linalg.pinv(square_roles_matrix)
    # dual_roles = filter_dual_roles(dual_roles)
    dual_roles_3d_shape = (1, number_roles, dual_roles.shape[1])
    reshaped_dual_roles = dual_roles[:number_roles].reshape(dual_roles_3d_shape)

    input_shapes = (tensor_representation_3d_shape, dual_roles_3d_shape)
    keras_decoder = keras_filler_decoder_network(input_shapes)

    with K.get_session():
        print('Running Keras encoder')

        reshaped_fillers = FILLERS_VECTORS.reshape(fillers_shape)
        reshaped_roles = final_processed_roles.reshape(roles_shape)

        tensor_representation = keras_encoder.predict_on_batch([
            reshaped_fillers,
            reshaped_roles
        ])

        print('Structural tensor representation')
        print(tensor_representation)

        print('Running Keras decoder')

        reshaped_tensor_representation = tensor_representation.reshape(tensor_representation_3d_shape)

        fillers_restored = keras_decoder.predict_on_batch([
            reshaped_tensor_representation,
            reshaped_dual_roles
        ])

    for i, filler in enumerate(fillers_restored):
        print('Original: '
              '\n\t[role]({}) with \t\t\t\t[filler]({}). '
              '\n\tDecoded: with [dual role]({})\t\t\t[filler]({})'
              .format(FILLER_ROLES_MAPPING[ORDER[i]], FILLERS_VECTORS[i], dual_roles[i], filler))
