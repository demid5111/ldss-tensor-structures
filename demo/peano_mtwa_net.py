from time import time

import numpy as np

from core.joiner.utils import generate_shapes
from core.joiner.vendor.network import build_tree_joiner_network
from core.peano.decode.vendor.network import build_decode_number_network
from core.peano.increment.vendor.network import build_increment_network
from core.peano.sum.vendor.network import build_sum_network
from core.peano.utils import number_to_tree, get_max_tree_depth
from core.utils import flattenize_per_tensor_representation
from demo.unshifting_structure import extract_per_level_tensor_representation_after_unshift


def encode(number, max_depth, roles, fillers, joiner_network=None):
    new_number_one = number_to_tree(number, max_depth, fillers, roles, joiner_network)
    return flattenize_per_tensor_representation(new_number_one)


def decode(network_layer, max_depth, dual_roles, fillers):
    single_role_shape = dual_roles[0].shape
    single_filler_shape = fillers[0].shape
    new_number_tree = extract_per_level_tensor_representation_after_unshift(network_layer,
                                                                            max_depth,
                                                                            single_role_shape,
                                                                            single_filler_shape)

    return decode_number(number_tree=new_number_tree,
                                  fillers=fillers,
                                  dual_roles=dual_roles,
                                  max_depth=max_depth)


def decode_number(number_tree, fillers, dual_roles, max_depth):
    is_zero = not np.any(flattenize_per_tensor_representation(number_tree))
    current_depth = max_depth
    current_number_tree = number_tree
    acc = 0
    while not is_zero:
        flattened_number = flattenize_per_tensor_representation(current_number_tree)
        if not np.any(flattenize_per_tensor_representation(number_tree)):
            break

        keras_number_decoder = build_decode_number_network(fillers=fillers,
                                                           dual_roles=dual_roles,
                                                           max_depth=current_depth)

        current_number_tree, is_not_zero_output = keras_number_decoder.predict_on_batch([
            *flattened_number
        ])

        acc += 1
        current_depth -= 1

        if not np.any(is_not_zero_output[0]):
            is_zero = True
            continue
    return acc


def sum_numbers(a, b, max_depth, roles, dual_roles, fillers, number_sum_blocks):
    single_role_shape = roles[0].shape
    single_filler_shape = fillers[0].shape
    fillers_shapes = generate_shapes(max_tree_depth=max_depth,
                                     role_shape=single_role_shape,
                                     filler_shape=single_filler_shape)
    joiner_network = build_tree_joiner_network(roles=roles, fillers_shapes=fillers_shapes)
    start = time()
    a_encoded = encode(a, max_depth, roles, fillers, joiner_network)
    en_a = time() - start
    print(f'Encoding {a} took {en_a} ms')
    start = time()
    b_encoded = encode(b, max_depth, roles, fillers, joiner_network)
    en_b = time() - start
    print(f'Encoding {b} took {en_b} ms')

    start = time()
    keras_sum_network = build_sum_network(roles, fillers, dual_roles, max_depth, number_sum_blocks=number_sum_blocks)
    gen = time() - start
    print(f'Generating Sum network took {gen} ms')

    start = time()
    decremented_number, incremented_number = keras_sum_network.predict_on_batch([
        a_encoded,
        b_encoded
    ])
    inf = time() - start
    print(f'Inference of Sum network took {inf} ms')

    start = time()
    c = decode(decremented_number, max_depth, dual_roles, fillers)
    dec_c = time() - start
    print(f'Decoding of {c} took {dec_c} ms')
    start = time()
    d = decode(incremented_number, max_depth, dual_roles, fillers)
    dec_d = time() - start
    print(f'Decoding of {d} took {dec_d} ms')
    return c, d, (en_a, en_b, gen, inf, dec_c, dec_d)


if __name__ == '__main__':
    # Input information
    fillers = np.array([
        [7, 0, 0, 0, 0],  # A
    ])

    roles = np.array([
        [10, 0],  # r_0
        [0, 5],  # r_1
    ])
    dual_basic_roles_case_1 = np.linalg.inv(roles)
    order_case_active = ['A', ]

    MAX_NUMBER = 4
    MAX_TREE_DEPTH = get_max_tree_depth(MAX_NUMBER)
    SINGLE_ROLE_SHAPE = roles[0].shape
    SINGLE_FILLER_SHAPE = fillers[0].shape

    # TPR-Inc Network
    a = 0
    new_number_one = number_to_tree(a, MAX_TREE_DEPTH, fillers, roles)
    one_unshifted = flattenize_per_tensor_representation(new_number_one)

    keras_increment_network = build_increment_network(roles=roles,
                                                      dual_roles=dual_basic_roles_case_1,
                                                      fillers=fillers,
                                                      max_depth=MAX_TREE_DEPTH)
    print('Built increment network')

    new_number = keras_increment_network.predict_on_batch([
        one_unshifted
    ])

    new_number_tree = extract_per_level_tensor_representation_after_unshift(new_number, MAX_TREE_DEPTH,
                                                                            SINGLE_ROLE_SHAPE,
                                                                            SINGLE_FILLER_SHAPE)

    result_number = decode_number(number_tree=new_number_tree,
                                  fillers=fillers,
                                  dual_roles=dual_basic_roles_case_1,
                                  max_depth=MAX_TREE_DEPTH)
    print('After incrementing {}, get {}'.format(a, result_number))

    # TPR-Sum Network
    MAX_NUMBER = 4
    MAX_TREE_DEPTH = get_max_tree_depth(MAX_NUMBER)

    a = 2
    b = 1
    c, d, times = sum_numbers(a, b, MAX_TREE_DEPTH, roles, dual_basic_roles_case_1, fillers, number_sum_blocks=2)
    print('After {} + {}, get {} + {}'.format(a, b, c, d))

    for i, j in ((1, 1), (2, 1), (3, 1), (4, 1), (10, 1), (100, 1), (1000, 1), (10000, 1)):
        NUM_ITER = 1
        total_times = [0, 0, 0, 0, 0, 0]
        for iter in range(NUM_ITER):
            c, d, times = sum_numbers(i, j, MAX_TREE_DEPTH, roles, dual_basic_roles_case_1, fillers,
                                      number_sum_blocks=4)
            print(f'Iter #{iter}')
            print('After {} + {}, get {} + {}'.format(i, j, c, d))
            del c
            del d
            for idx in range(len(total_times)):
                total_times[idx] += times[idx]
        avg_times = tuple(t / NUM_ITER for t in total_times)
        print(f'Averages for {i} and {j}:')
        print(f' encoding a:  {avg_times[0]}')
        print(f' encoding b:  {avg_times[1]}')
        print(f' generating:  {avg_times[2]}')
        print(f' inference:  {avg_times[3]}')
        print(f' decoding c:  {avg_times[4]}')
        print(f' decoding d:  {avg_times[5]}')

