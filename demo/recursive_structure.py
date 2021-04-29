from functools import reduce

import numpy as np
import tensorflow as tf

from core.encoder.vendor.network import build_encoder_network as keras_encoder_network, \
    prepare_shapes as prepare_encoder_shapes
from core.decoder.vendor.network import build_filler_decoder_network as keras_filler_decoder_network, \
    prepare_shapes as prepare_decoder_shapes


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


def pre_process_roles(roles_vectors, mapping, order):
    accumulated_roles = accumulate_roles(roles_vectors, mapping)
    flattened_roles = flatten_roles(accumulated_roles)
    equal_length_roles = equalize_roles_length(flattened_roles)
    return np.array([equal_length_roles[i] for i in order])


def pre_process_input_data(roles_vectors, fillers_vectors, filler_role_mapping, filler_order):
    maximum_depth = find_maximum_depth(filler_role_mapping)
    number_fillers, dim_fillers = fillers_vectors.shape
    number_basic_roles, dim_roles = roles_vectors.shape

    assert number_fillers == dim_fillers, 'Fillers should be a quadratic matrix'
    assert number_basic_roles == dim_roles, 'Roles should be a quadratic matrix'

    tr_shape = calculate_tensor_representation_shape_as_2d_matrix(dim_fillers, dim_roles, maximum_depth)

    final_roles = pre_process_roles(roles_vectors, filler_role_mapping, filler_order)

    dual_basic_roles = np.linalg.inv(roles_vectors)
    final_dual_roles = pre_process_roles(dual_basic_roles, filler_role_mapping, filler_order)

    return final_roles, final_dual_roles, (1, *tr_shape)


def pre_process_and_run(roles_vectors, fillers_vectors, filler_role_mapping, filler_order):
    # Processing logic
    final_roles, final_dual_roles, tr_shape = pre_process_input_data(roles_vectors=roles_vectors,
                                                                     fillers_vectors=fillers_vectors,
                                                                     filler_role_mapping=filler_role_mapping,
                                                                     filler_order=filler_order)

    print('Building Keras encoder')
    fillers_shape, roles_shape = prepare_encoder_shapes(fillers_vectors, final_roles)
    keras_encoder = keras_encoder_network(input_shapes=(fillers_shape, roles_shape))

    print('Building Keras decoder')
    dual_roles_shape = prepare_decoder_shapes(mapping=filler_role_mapping,
                                              dual_roles=final_dual_roles)
    keras_decoder = keras_filler_decoder_network(input_shapes=(tr_shape, dual_roles_shape))

    print('Running Keras encoder')
    reshaped_fillers = fillers_vectors.reshape(fillers_shape)
    reshaped_roles = final_roles.reshape(roles_shape)

    tensor_representation = keras_encoder.predict_on_batch([
        reshaped_fillers,
        reshaped_roles
    ])

    print('Running Keras decoder')
    reshaped_dual_roles = final_dual_roles.reshape(dual_roles_shape)

    reshaped_tensor_representation = tensor_representation.reshape(tr_shape)

    fillers_restored = keras_decoder.predict_on_batch([
        reshaped_tensor_representation,
        reshaped_dual_roles
    ])
    return tensor_representation, fillers_restored, final_dual_roles


def print_results(fillers_restored, fillers_vectors, filler_role_mapping, filler_order, final_dual_roles):
    for i, filler in enumerate(fillers_restored):
        print('Original: '
              '\n\t[role]({}) with \t\t\t\t[filler]({}). '
              '\n\tDecoded: with [dual role]({})\t\t\t[filler]({})'
              .format(filler_role_mapping[filler_order[i]], fillers_vectors[i], final_dual_roles[i], filler))


if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    """
    First use case for the structure that has nesting equal 1

    root
    | \  \
    A B  C
    """
    # Input information
    fillers_case_1 = np.array([
        [7, 0, 0],  # A
        [0, 13, 0],  # B
        [0, 0, 2],  # C
    ])
    roles_case_1 = np.array([
        [10, 0, 0],  # r_0
        [0, 5, 0],  # r_1
        [0, 0, 8],  # r_2
    ])
    mapping_case_1 = {
        'A': [0],
        'B': [1],
        'C': [2]
    }
    order_case_1 = ['A', 'B', 'C']

    """
    Second use case for the structure that has variable nesting

    Using this structure for demonstration:
    root
    |  \
    A  / \
      B   \
         / \
        C  D
    """
    # Input information
    fillers_case_2 = np.array([
        [8, 0, 0, 0],  # A
        [0, 15, 0, 0],  # B
        [0, 0, 10, 0],  # C
        [0, 0, 0, 3],  # D
    ])
    roles_case_2 = np.array([
        [10, 0],  # r_0
        [0, 5],  # r_1
    ])
    mapping_case_2 = {
        'A': [0],
        'B': [0, 1],
        'C': [0, 1, 1],
        'D': [1, 1, 1]
    }
    order_case_2 = ['A', 'B', 'C', 'D']

    # with K.get_session():
    tr_1, fillers_restored_case_1, final_dual_roles_case_1 = pre_process_and_run(roles_vectors=roles_case_1,
                                                                                 fillers_vectors=fillers_case_1,
                                                                                 filler_role_mapping=mapping_case_1,
                                                                                 filler_order=order_case_1)
    print_results(fillers_restored=fillers_restored_case_1,
                  fillers_vectors=fillers_case_1,
                  filler_role_mapping=mapping_case_1,
                  filler_order=order_case_1,
                  final_dual_roles=final_dual_roles_case_1)

    tr_2, fillers_restored_case_2, final_dual_roles_case_2 = pre_process_and_run(roles_vectors=roles_case_2,
                                                                                 fillers_vectors=fillers_case_2,
                                                                                 filler_role_mapping=mapping_case_2,
                                                                                 filler_order=order_case_2)

    print_results(fillers_restored=fillers_restored_case_2,
                  fillers_vectors=fillers_case_2,
                  filler_role_mapping=mapping_case_2,
                  filler_order=order_case_2,
                  final_dual_roles=final_dual_roles_case_2)
