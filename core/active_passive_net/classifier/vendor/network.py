import tensorflow as tf

from core.joiner.vendor.network import constant_input, mat_mul
from core.unshifter.vendor.network import unshift_matrix


def normalization(x):
    return tf.keras.backend.switch(x > 0., x / x, tf.keras.backend.zeros_like(x))


def build_one_level_extraction_branch(model_input, roles, filler_len, max_depth, stop_level, role_extraction_order,
                                      prefix=''):
    return build_universal_extraction_branch(model_input, roles, filler_len, max_depth, stop_level,
                                             role_extraction_order, prefix=prefix)


def build_universal_extraction_branch(model_input, roles, filler_len, max_depth, stop_level, role_extraction_order,
                                      prefix=''):
    shift_inputs = []
    current_input = model_input
    target_num_elements = 0
    for level_index, role_index in zip(range(max_depth, stop_level - 1, -1), role_extraction_order):
        _, flattened_num_elements = unshift_matrix(roles[role_index], filler_len, level_index).shape
        layer_name = 'constant_input_level_{}_(ex{})'.format(prefix + '_' if prefix else '', level_index, role_index)
        left_shift_input = constant_input(roles[role_index], filler_len, level_index, layer_name, unshift_matrix)
        shift_inputs.append(left_shift_input)

        current_num_elements = flattened_num_elements + filler_len
        target_num_elements = flattened_num_elements

        # TODO: resolve custom reshape issue
        reshape_for_crop = tf.keras.layers.Lambda(lambda x: tf.keras.backend.reshape(x, (1, current_num_elements, 1)))(
            current_input)
        clip_first_level = tf.keras.layers.Cropping1D(cropping=(filler_len, 0))(reshape_for_crop)
        current_input = tf.keras.layers.Lambda(lambda x: tf.keras.backend.reshape(x, (target_num_elements, 1)))(
            clip_first_level)

        current_input = tf.keras.layers.Lambda(mat_mul)([
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
    flattened_tree_input = tf.keras.layers.Input(shape=(*shape,))

    extraction_inputs, extraction_output, _ = build_universal_extraction_branch(model_input=flattened_tree_input,
                                                                                roles=roles,
                                                                                filler_len=filler_len,
                                                                                max_depth=max_depth,
                                                                                stop_level=stop_level,
                                                                                role_extraction_order=role_extraction_order)
    reshape_for_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.reshape(x, (1, filler_len, 1)))(
        extraction_output)
    global_max_pool = tf.keras.layers.GlobalMaxPooling1D()(reshape_for_pool)
    normalizer = tf.keras.layers.Lambda(normalization)(global_max_pool)
    return extraction_inputs, flattened_tree_input, normalizer, extraction_output


def build_filler_extractor_network(roles, fillers, tree_shape, role_extraction_order, stop_level=0):
    const_inputs, variable_input, output, _ = build_classification_branch(roles=roles,
                                                                          fillers=fillers,
                                                                          tree_shape=tree_shape,
                                                                          role_extraction_order=role_extraction_order,
                                                                          stop_level=stop_level)
    return tf.keras.Model(
        inputs=[
            *const_inputs,
            variable_input,
        ],
        outputs=output)


# TODO: rename to correspond to classification
def build_real_filler_extractor_network(roles, fillers, tree_shape, role_extraction_order, stop_level=0):
    filler_len = fillers[0].shape[0]
    max_depth = len(tree_shape) - 1
    _, flattened_tree_num_elements = unshift_matrix(roles[0], filler_len, max_depth).shape
    shape = (flattened_tree_num_elements + filler_len, 1)
    flattened_tree_input = tf.keras.layers.Input(shape=(*shape,), batch_size=1)

    extraction_inputs, extraction_output, _ = build_universal_extraction_branch(model_input=flattened_tree_input,
                                                                                roles=roles,
                                                                                filler_len=filler_len,
                                                                                max_depth=max_depth,
                                                                                stop_level=stop_level,
                                                                                role_extraction_order=role_extraction_order)
    return tf.keras.Model(
        inputs=[
            *extraction_inputs,
            flattened_tree_input
        ],
        outputs=extraction_output)
