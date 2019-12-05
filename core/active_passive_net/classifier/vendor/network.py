from keras.engine import Input
from keras.layers import Lambda, Cropping1D, GlobalMaxPooling1D
from keras.models import Model
import keras.backend as K

from core.joiner.vendor.network import constant_input, mat_mul
from core.unshifter.vendor.network import unshift_matrix


def normalization(x):
    return K.switch(x > 0., x / x, K.zeros_like(x))


def build_one_level_extraction_branch(model_input, roles, filler_len, max_depth, stop_level, role_extraction_order):
    return build_universal_extraction_branch(model_input, roles, filler_len, max_depth, stop_level,
                                             role_extraction_order)


def build_universal_extraction_branch(model_input, roles, filler_len, max_depth, stop_level, role_extraction_order):
    shift_inputs = []
    current_input = model_input
    target_num_elements = 0
    for level_index, role_index in zip(range(max_depth, stop_level - 1, -1), role_extraction_order):
        _, flattened_num_elements = unshift_matrix(roles[role_index], filler_len, level_index).shape
        layer_name = 'constant_input_level_{}_(ex{})'.format(level_index, role_index)
        left_shift_input = constant_input(roles[role_index], filler_len, level_index, layer_name, unshift_matrix)
        shift_inputs.append(left_shift_input)

        current_num_elements = flattened_num_elements + filler_len
        target_num_elements = flattened_num_elements

        # TODO: resolve custom reshape issue
        reshape_for_crop = Lambda(lambda x: K.tf.reshape(x, (1, current_num_elements, 1)))(current_input)
        clip_first_level = Cropping1D(cropping=(filler_len, 0))(reshape_for_crop)
        current_input = Lambda(lambda x: K.tf.reshape(x, (target_num_elements, 1)))(clip_first_level)

        current_input = Lambda(mat_mul)([
            left_shift_input,
            current_input
        ])
    return shift_inputs, current_input, target_num_elements


def build_classification_branch(roles, fillers, tree_shape, role_extraction_order, stop_level=0):
    filler_len = fillers[0].shape[0]
    max_depth = len(tree_shape) - 1
    assert max_depth == len(role_extraction_order), 'Extraction should happen until the final filler'

    _, flattened_tree_num_elements = unshift_matrix(roles[0], filler_len, max_depth).shape
    shape = (flattened_tree_num_elements + filler_len, 1)
    flattened_tree_input = Input(shape=(*shape,), batch_shape=(*shape,))

    extraction_inputs, extraction_output, _ = build_universal_extraction_branch(model_input=flattened_tree_input,
                                                                                roles=roles,
                                                                                filler_len=filler_len,
                                                                                max_depth=max_depth,
                                                                                stop_level=stop_level,
                                                                                role_extraction_order=role_extraction_order)
    reshape_for_pool = Lambda(lambda x: K.tf.reshape(x, (1, filler_len, 1)))(extraction_output)
    global_max_pool = GlobalMaxPooling1D()(reshape_for_pool)
    normalizer = Lambda(normalization)(global_max_pool)
    return extraction_inputs, flattened_tree_input, normalizer


def build_filler_extractor_network(roles, fillers, tree_shape, role_extraction_order, stop_level=0):
    const_inputs, variable_input, output = build_classification_branch(roles=roles,
                                                                       fillers=fillers,
                                                                       tree_shape=tree_shape,
                                                                       role_extraction_order=role_extraction_order,
                                                                       stop_level=stop_level)
    return Model(
        inputs=[
            *const_inputs,
            variable_input,
        ],
        outputs=output)
