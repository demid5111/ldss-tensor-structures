import tensorflow as tf

from core.active_passive_net.active_extractor.vendor.network import custom_constant_layer
from core.peano.increment.vendor.network import constant_inputs_for_increment_block, increment_block, condition_branch, \
    build_extract_branch
from core.unshifter.vendor.network import unshift_matrix


def sum_block(incrementing_input, decrementing_input,
              increment_value, roles, dual_roles, filler_len, max_depth, block_id,
              left_shift_input, right_shift_input, constant_input_filler, constant_for_decrementing_input):
    output = increment_block(
        incrementing_input=incrementing_input,
        increment_value=increment_value,
        roles=roles,
        dual_roles=dual_roles,
        filler_len=filler_len,
        max_depth=max_depth,
        block_id=block_id,
        left_shift_input=left_shift_input,
        right_shift_input=right_shift_input,
        constant_input_filler=constant_input_filler
    )

    incremented_output, _ = condition_branch(
        condition_input=decrementing_input,
        condition_if_not_zero=output,
        condition_if_zero=incrementing_input,
        dual_roles=dual_roles,
        filler_len=filler_len,
        max_depth=max_depth,
        block_id=block_id + 2
    )

    decremented_input = build_extract_branch(
        input_layer=decrementing_input,
        extract_role=dual_roles[0],
        filler_len=filler_len,
        max_depth=max_depth - 1,
        block_id=block_id + 4
    )

    decremented_output = tf.keras.layers.Concatenate(axis=1)([decremented_input, constant_for_decrementing_input])

    return incremented_output, decremented_output


def build_sum_network(roles, fillers, dual_roles, max_depth, number_sum_blocks=1):
    filler_len = fillers[0].shape[0]

    input_num_elements, flattened_tree_num_elements = unshift_matrix(roles[0], filler_len, max_depth - 1).shape
    shape = (flattened_tree_num_elements + filler_len, 1)
    flattened_decrementing_input = tf.keras.layers.Input(shape=(*shape,), batch_size=1, name='left_operand')
    flattened_incrementing_input = tf.keras.layers.Input(shape=(*shape,), batch_size=1, name='right_operand')

    block_id = 0
    shift_input, tmp_reshaped_increment, tmp_reshaped_fake_filler = constant_inputs_for_increment_block(roles, fillers, max_depth,
                                                                                     block_id)
    left_shift_input, right_shift_input = shift_input

    target_elements, _ = unshift_matrix(roles[0], filler_len, max_depth - 1).shape
    constant_for_decrementing_input = custom_constant_layer(const_size=target_elements + filler_len, name='const_one')

    incremented = flattened_incrementing_input
    decremented = flattened_decrementing_input
    for i in range(number_sum_blocks):
        incremented, decremented = sum_block(
            incrementing_input=incremented,
            decrementing_input=decremented,
            increment_value=tmp_reshaped_increment,
            roles=roles,
            dual_roles=dual_roles,
            filler_len=filler_len,
            max_depth=max_depth,
            block_id=block_id + i * 5,
            left_shift_input=left_shift_input,
            right_shift_input=right_shift_input,
            constant_input_filler=tmp_reshaped_fake_filler,
            constant_for_decrementing_input=constant_for_decrementing_input)

    return tf.keras.Model(
        inputs=[
            flattened_decrementing_input,
            flattened_incrementing_input,
        ],
        outputs=[
            decremented,
            incremented
        ])
