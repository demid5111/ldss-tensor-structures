from keras import Input, Model
from keras.layers import Concatenate

from core.active_passive_net.active_extractor.vendor.network import custom_constant_layer
from core.peano.increment.vendor.network import constant_inputs_for_increment_block, increment_block, condition_branch, \
    build_extract_branch
from core.unshifter.vendor.network import unshift_matrix


def sum_block(incrementing_input, decrementing_input,
              increment_value, roles, dual_roles, filler_len, max_depth, block_id,
              left_shift_input, right_shift_input, constant_input_filler, constant_for_decrementing_input):
    increment_const_inputs, output = increment_block(
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

    const_condition_inputs, incremented_output, _ = condition_branch(
        condition_input=decrementing_input,
        condition_if_not_zero=output,
        condition_if_zero=incrementing_input,
        dual_roles=dual_roles,
        filler_len=filler_len,
        max_depth=max_depth,
        block_id=block_id + 2
    )

    const_extract_inputs, decremented_input = build_extract_branch(
        input_layer=decrementing_input,
        extract_role=dual_roles[0],
        filler_len=filler_len,
        max_depth=max_depth - 1,
        block_id=block_id + 4
    )

    decremented_output = Concatenate(axis=0)([decremented_input, constant_for_decrementing_input])

    return (
               *increment_const_inputs,
               *const_condition_inputs,
               *const_extract_inputs
           ), incremented_output, decremented_output


def build_sum_network(roles, fillers, dual_roles, max_depth, number_sum_blocks=1):
    filler_len = fillers[0].shape[0]

    input_num_elements, flattened_tree_num_elements = unshift_matrix(roles[0], filler_len, max_depth - 1).shape
    shape = (flattened_tree_num_elements + filler_len, 1)
    flattened_decrementing_input = Input(shape=(*shape,), batch_shape=(*shape,), name='left_operand')
    flattened_incrementing_input = Input(shape=(*shape,), batch_shape=(*shape,), name='right_operand')

    block_id = 0
    shift_input, increment_input, filler_input = constant_inputs_for_increment_block(roles, fillers, max_depth,
                                                                                     block_id)
    left_shift_input, right_shift_input = shift_input
    tmp_reshaped_increment, const_increment = increment_input
    tmp_reshaped_fake_filler, const_filler = filler_input

    target_elements, _ = unshift_matrix(roles[0], filler_len, max_depth - 1).shape
    tmp_reshaped_fake, const_one = custom_constant_layer(const_size=target_elements + filler_len, name='const_one')

    all_sum_const_inputs = []
    incremented = flattened_incrementing_input
    decremented = flattened_decrementing_input
    for i in range(number_sum_blocks):
        sum_const_inputs, incremented, decremented = sum_block(
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
            constant_for_decrementing_input=tmp_reshaped_fake)
        all_sum_const_inputs.extend(sum_const_inputs)

    return Model(
        inputs=[
            left_shift_input,
            right_shift_input,
            const_increment,
            const_filler,
            const_one,
            *all_sum_const_inputs,
            flattened_decrementing_input,
            flattened_incrementing_input,
        ],
        outputs=[
            decremented,
            incremented
        ])
