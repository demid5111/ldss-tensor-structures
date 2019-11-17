import numpy as np

from keras.engine import Input
from keras.layers import Lambda, Permute, Cropping1D, Reshape
from keras.models import Model
import keras.backend as K

from core.joiner.vendor.network import constant_input, mat_mul
from core.unshifter.vendor.network import unshift_matrix


# def mat_mul(tensors):
#     return [K.dot(tensors[0],tensors[1][i]) for i in range(tensors[1].shape[0])]

def build_filler_extractor_network(roles, fillers, tree_shape, role_extraction_order, stop_level=0):
    filler_len = fillers[0].shape[0]
    max_depth = len(tree_shape) - 1

    _, flattened_tree_num_elements = unshift_matrix(roles[0], filler_len, max_depth).shape
    shape = (flattened_tree_num_elements + filler_len, 1)
    flattened_tree_input = Input(shape=(*shape,), batch_shape=(*shape,))

    shift_inputs = []
    current_input = None
    for level_index, role_index in zip(range(max_depth, stop_level - 1, -1), role_extraction_order):
        _, flattened_num_elements = unshift_matrix(roles[0], filler_len, level_index).shape
        layer_name = 'constant_input_level_{}_(ex{})'.format(level_index, role_index)
        left_shift_input = constant_input(roles[role_index], filler_len, level_index, layer_name, unshift_matrix)
        shift_inputs.append(left_shift_input)

        if current_input is None:
            current_num_elements = flattened_tree_num_elements + filler_len
            target_num_elements = flattened_num_elements
            current_input = flattened_tree_input
        else:
            current_num_elements = flattened_num_elements + filler_len
            target_num_elements = flattened_num_elements

        # TODO: resolve custom reshape issue
        reshape_for_crop = Lambda(lambda x: K.tf.reshape(x, (1, current_num_elements, 1)))(
            current_input if current_input is not None else flattened_tree_input)
        clip_first_level = Cropping1D(cropping=(filler_len, 0))(reshape_for_crop)
        current_input = Lambda(lambda x: K.tf.reshape(x, (target_num_elements, 1)))(clip_first_level)

        current_input = Lambda(mat_mul)([
            left_shift_input,
            current_input
        ])

    return Model(
        inputs=[
            *shift_inputs,
            flattened_tree_input,
        ],
        outputs=current_input)
