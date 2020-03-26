import keras.backend as K
import numpy as np
from keras import Model, Input
from keras.layers import Concatenate, Lambda, GlobalMaxPooling1D, Add, Multiply

from core.active_passive_net.active_extractor.vendor.network import custom_cropping_layer, custom_constant_layer
from core.active_passive_net.classifier.vendor.network import build_one_level_extraction_branch, normalization
from core.active_passive_net.passive_extractor.vendor.network import build_join_branch
from core.unshifter.vendor.network import unshift_matrix


def make_output_same_length_as_input(layer_to_crop, role, filler_len, max_depth):
    target_num_elements, flattened_num_elements = unshift_matrix(role, filler_len, max_depth).shape
    return custom_cropping_layer(
        input_layer=layer_to_crop,
        crop_from_beginning=0,
        crop_from_end=flattened_num_elements + filler_len - target_num_elements,
        input_tensor_length=flattened_num_elements + filler_len,
        final_tensor_length=target_num_elements
    )


def build_extract_branch(input_layer, extract_role, filler_len, max_depth, branch_id=1):
    stop_level_for_one = max_depth - 1
    one_branch = build_one_level_extraction_branch(model_input=input_layer,
                                                   roles=[None, extract_role],
                                                   filler_len=filler_len,
                                                   max_depth=max_depth,
                                                   stop_level=stop_level_for_one,
                                                   role_extraction_order=[1],
                                                   prefix='single_extract_{}'.format(branch_id))
    one_extraction_const_inputs, one_raw_output, _ = one_branch
    return [
               *one_extraction_const_inputs,
           ], one_raw_output


def check_if_not_zero_branch(decrementing_input, role, filler_len, max_depth, block_id):
    const_inputs, one_tensor_output = build_extract_branch(
        input_layer=decrementing_input,
        extract_role=role,
        filler_len=filler_len,
        max_depth=max_depth - 1,
        branch_id=block_id
    )

    target_elements, _ = unshift_matrix(role, filler_len, max_depth - 1).shape
    reshape_for_pool = Lambda(lambda x: K.tf.reshape(x, (1, target_elements, 1)))(one_tensor_output)
    global_max_pool = GlobalMaxPooling1D()(reshape_for_pool)
    return const_inputs, Lambda(normalization)(global_max_pool)


def check_if_zero_branch(flag_input):
    np_constant = np.array([-1])
    tf_constant = K.constant(np_constant)
    const_neg_1 = Input(tensor=tf_constant, shape=np_constant.shape, dtype='int32',
                        name='increment_neg_1')

    sum_is_zero_const = Add()([flag_input, const_neg_1])
    return const_neg_1, Multiply()([sum_is_zero_const, const_neg_1])


def single_sum_block(decrementing_input, incrementing_input, constant_input_one, constant_input_filler, roles,
                     filler_len, dual_roles, max_depth, increment_value):
    block_id = 1

    next_number_const_inputs, next_number = build_join_branch(
        roles=roles,
        filler_len=filler_len,
        max_depth=max_depth,
        inputs=[
            incrementing_input,
            increment_value
        ],
        prefix='cons_'.format(block_id)
    )
    next_number_output = Concatenate(axis=0)([constant_input_filler, next_number])

    const_first_operand_inputs, is_first_operand_not_zero = check_if_not_zero_branch(
        decrementing_input=decrementing_input,
        role=dual_roles[1],
        filler_len=filler_len,
        max_depth=max_depth,
        block_id=block_id + 1)
    const_second_operand_inputs, is_second_operand_not_zero = check_if_not_zero_branch(
        decrementing_input=incrementing_input,
        role=dual_roles[1],
        filler_len=filler_len,
        max_depth=max_depth,
        block_id=block_id + 2)
    is_both_not_zero = Multiply()([is_first_operand_not_zero, is_second_operand_not_zero])
    is_first_operand_not_zero_branch = Lambda(lambda tensors: tensors[0] * tensors[1])(
        [next_number_output, is_both_not_zero])

    const_input, is_any_zero = check_if_zero_branch(is_both_not_zero)
    lengthened_input = Concatenate(axis=0)([increment_value, constant_input_one])
    is_any_zero_branch = Lambda(lambda tensors: tensors[0] * tensors[1])(
        [lengthened_input, is_any_zero])

    sum_branches = Add()([is_any_zero_branch, is_first_operand_not_zero_branch])

    cropped_number = make_output_same_length_as_input(layer_to_crop=sum_branches,
                                                      role=dual_roles[1],
                                                      filler_len=filler_len,
                                                      max_depth=max_depth)
    return [
               *const_first_operand_inputs,
               *const_second_operand_inputs,
               const_input,
               *next_number_const_inputs
           ], cropped_number


def build_increment_network(roles, fillers, dual_roles, max_depth, increment_value):
    filler_len = fillers[0].shape[0]

    input_num_elements, flattened_tree_num_elements = unshift_matrix(roles[0], filler_len, max_depth - 1).shape
    shape = (flattened_tree_num_elements + filler_len, 1)
    flattened_decrementing_input = Input(shape=(*shape,), batch_shape=(*shape,))
    flattened_incrementing_input = Input(shape=(*shape,), batch_shape=(*shape,))

    target_elements, _ = unshift_matrix(roles[0], filler_len, max_depth).shape
    # later we have to join two subtrees of different depth. for that we have to
    # make filler of verb of the same depth - make fake constant layer
    tmp_reshaped_fake, const_one = custom_constant_layer(const_size=target_elements + filler_len, name='const_one')

    tmp_reshaped_fake_filler, const_filler = custom_constant_layer(const_size=filler_len, name='const_filler')
    tmp_reshaped_increment, const_increment = custom_constant_layer(const_size=filler_len,
                                                                    name='const_increment',
                                                                    np_constant=increment_value)

    inputs, output = single_sum_block(decrementing_input=flattened_decrementing_input,
                                      incrementing_input=flattened_incrementing_input,
                                      constant_input_one=tmp_reshaped_fake,
                                      constant_input_filler=tmp_reshaped_fake_filler,
                                      roles=roles,
                                      filler_len=filler_len,
                                      dual_roles=dual_roles,
                                      max_depth=max_depth,
                                      increment_value=tmp_reshaped_increment)

    return Model(
        inputs=[
            flattened_decrementing_input,
            flattened_incrementing_input,
            *inputs,
            const_one,
            const_filler,
            const_increment
        ],
        outputs=output)
