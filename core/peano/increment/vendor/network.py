import numpy as np
import tensorflow as tf

from core.active_passive_net.active_extractor.vendor.network import custom_cropping_layer
from core.active_passive_net.classifier.vendor.network import build_one_level_extraction_branch, normalization
from core.active_passive_net.passive_extractor.vendor.network import build_join_branch, create_shift_matrix_as_input
from core.peano.utils import number_to_tree
from core.unshifter.vendor.network import unshift_matrix
from core.utils import flattenize_per_tensor_representation, create_custom_constant


def make_output_same_length_as_input(layer_to_crop, role, filler_len, max_depth):
    target_num_elements, flattened_num_elements = unshift_matrix(role, filler_len, max_depth).shape
    return custom_cropping_layer(
        input_layer=layer_to_crop,
        crop_from_beginning=0,
        crop_from_end=flattened_num_elements + filler_len - target_num_elements,
        input_tensor_length=flattened_num_elements + filler_len,
        final_tensor_length=target_num_elements
    )


def build_extract_branch(input_layer, extract_role, filler_len, max_depth, block_id):
    stop_level_for_one = max_depth - 1
    one_branch = build_one_level_extraction_branch(model_input=input_layer,
                                                   roles=[None, extract_role],
                                                   filler_len=filler_len,
                                                   max_depth=max_depth,
                                                   stop_level=stop_level_for_one,
                                                   role_extraction_order=[1],
                                                   prefix='single_extract_{}'.format(block_id))
    one_raw_output, _ = one_branch
    return one_raw_output


def check_if_not_zero_branch(decrementing_input, role, filler_len, max_depth, block_id):
    one_tensor_output = build_extract_branch(
        input_layer=decrementing_input,
        extract_role=role,
        filler_len=filler_len,
        max_depth=max_depth - 1,
        block_id=block_id
    )

    target_elements, _ = unshift_matrix(role, filler_len, max_depth - 1).shape
    reshape_for_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.reshape(x, (1, target_elements, 1)))(
        one_tensor_output)
    global_max_pool = tf.keras.layers.GlobalMaxPooling1D()(reshape_for_pool)
    return tf.keras.layers.Lambda(normalization)(global_max_pool), one_tensor_output


def check_if_zero_branch(flag_input):
    const_neg_1 = tf.constant(np.array([-1]), dtype='float32')
    sum_is_zero_const = tf.keras.layers.Add()([flag_input, const_neg_1])
    return tf.keras.layers.Multiply()([sum_is_zero_const, const_neg_1])


def condition_branch(condition_input, condition_if_not_zero, condition_if_zero, dual_roles, filler_len, max_depth,
                     block_id):
    # TODO: generalize for a APNet classification branch
    is_not_zero, decremented_input = check_if_not_zero_branch(
        decrementing_input=condition_input,
        role=dual_roles[1],
        filler_len=filler_len,
        max_depth=max_depth,
        block_id=block_id + 1)

    is_zero = check_if_zero_branch(is_not_zero)
    is_value_zero_branch = tf.keras.layers.Lambda(lambda tensors: tensors[0] * tensors[1])([
        condition_if_zero,
        is_zero
    ])

    is_value_not_zero_branch = tf.keras.layers.Lambda(lambda tensors: tensors[0] * tensors[1])([
        condition_if_not_zero,
        is_not_zero
    ])
    sum_branches = tf.keras.layers.Add()([is_value_zero_branch, is_value_not_zero_branch])
    return sum_branches, decremented_input


def increment_block(incrementing_input, increment_value, roles, dual_roles, filler_len, max_depth, block_id,
                    left_shift_input,
                    right_shift_input, constant_input_filler):
    next_number = build_join_branch(
        roles=roles,
        filler_len=filler_len,
        max_depth=max_depth,
        inputs=[
            incrementing_input,
            increment_value
        ],
        prefix='cons_'.format(block_id),
        left_shift_input=left_shift_input,
        right_shift_input=right_shift_input
    )
    next_number_reshaped = tf.keras.layers.Lambda(lambda x: tf.keras.backend.reshape(x, (1, *next_number.shape)))(
        next_number)
    next_number_output = tf.keras.layers.Concatenate(axis=1)([constant_input_filler, next_number_reshaped])
    cropped_number_after_increment = make_output_same_length_as_input(layer_to_crop=next_number_output,
                                                                      role=roles[1],
                                                                      filler_len=filler_len,
                                                                      max_depth=max_depth)

    output, _ = condition_branch(
        condition_input=incrementing_input,
        condition_if_not_zero=cropped_number_after_increment,
        condition_if_zero=increment_value,
        dual_roles=dual_roles,
        filler_len=filler_len,
        max_depth=max_depth,
        block_id=block_id
    )
    return output


def constant_inputs_for_increment_block(roles, fillers, max_depth, block_id):
    filler_len = fillers[0].shape[0]

    # Joining matrices
    prefix = '{}_increment_block_'.format(block_id)
    left_shift_input = create_shift_matrix_as_input(roles[0], 0, filler_len, max_depth, prefix)
    right_shift_input = create_shift_matrix_as_input(roles[1], 1, filler_len, max_depth, prefix)

    # Incrementing value
    new_number_one = number_to_tree(1, max_depth, fillers, roles)
    one = flattenize_per_tensor_representation(new_number_one)
    np_constant = create_custom_constant(const_size=filler_len, np_constant=one)

    # Filler constant for filling first level that is missed after join
    tmp_reshaped_fake_filler = create_custom_constant(const_size=filler_len)

    return (
               left_shift_input,
               right_shift_input
           ), np_constant, (
               tmp_reshaped_fake_filler
           )


def build_increment_network(roles, dual_roles, fillers, max_depth):
    filler_len = fillers[0].shape[0]

    # Number to be incremented
    input_num_elements, flattened_tree_num_elements = unshift_matrix(roles[0], filler_len, max_depth - 1).shape
    shape = (flattened_tree_num_elements + filler_len, 1)
    flattened_incrementing_input = tf.keras.layers.Input(shape=(*shape,), batch_size=1)

    block_id = 0
    shift_input, tmp_reshaped_increment, tmp_reshaped_fake_filler = constant_inputs_for_increment_block(roles, fillers, max_depth,
                                                                                     block_id)
    left_shift_input, right_shift_input = shift_input

    output = increment_block(
        incrementing_input=flattened_incrementing_input,
        increment_value=tmp_reshaped_increment,
        roles=roles,
        dual_roles=dual_roles,
        filler_len=filler_len,
        max_depth=max_depth,
        block_id=block_id,
        left_shift_input=left_shift_input,
        right_shift_input=right_shift_input,
        constant_input_filler=tmp_reshaped_fake_filler
    )

    return tf.keras.Model(
        inputs=[
            flattened_incrementing_input,
        ],
        outputs=output)
