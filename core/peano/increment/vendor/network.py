import keras.backend as K
import numpy as np
from keras import Model, Input
from keras.layers import Lambda, Concatenate

from core.active_passive_net.active_extractor.vendor.network import custom_cropping_layer
from core.active_passive_net.classifier.vendor.network import build_one_level_extraction_branch
from core.active_passive_net.passive_extractor.vendor.network import build_join_branch
from core.unshifter.vendor.network import unshift_matrix


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


def single_sum_block(decrementing_input, incrementing_input, constant_input, roles, filler_len, dual_roles, max_depth):
    block_id = 1

    const_inputs, one_tensor_output = build_extract_branch(
        input_layer=decrementing_input,
        extract_role=dual_roles[1],
        filler_len=filler_len,
        max_depth=max_depth - 1,
        branch_id=block_id
    )

    concatenate_one = Concatenate(axis=0)([one_tensor_output, constant_input])

    next_number_const_inputs, next_number_output = build_join_branch(
        roles=roles,
        filler_len=filler_len,
        max_depth=max_depth,
        inputs=[
            incrementing_input,
            concatenate_one
        ],
        prefix='cons_'.format(block_id)
    )

    target_num_elements, flattened_num_elements = unshift_matrix(dual_roles[1], filler_len, max_depth).shape
    cropped_number = custom_cropping_layer(
        input_layer=next_number_output,
        crop_from_beginning=0,
        crop_from_end=flattened_num_elements - target_num_elements,
        input_tensor_length=flattened_num_elements,
        final_tensor_length=target_num_elements
    )
    return [
               *const_inputs,
               *next_number_const_inputs
           ], cropped_number


def build_increment_network(roles, fillers, dual_roles, max_depth):
    filler_len = fillers[0].shape[0]

    input_num_elements, flattened_tree_num_elements = unshift_matrix(roles[0], filler_len, max_depth - 1).shape
    shape = (flattened_tree_num_elements + filler_len, 1)
    flattened_decrementing_input = Input(shape=(*shape,), batch_shape=(*shape,))
    flattened_incrementing_input = Input(shape=(*shape,), batch_shape=(*shape,))

    # _, flattened_tree_num_elements_extender = unshift_matrix(roles[0], filler_len, max_depth-1).shape
    # later we have to join two subtrees of different depth. for that we have to
    # make filler of verb of the same depth - make fake constant layer
    fake_size = input_num_elements + filler_len
    np_constant = np.zeros((fake_size, 1))
    tf_constant = K.constant(np_constant)
    const_fake_extender = Input(tensor=tf_constant, batch_shape=np_constant.shape, shape=np_constant.shape,
                                dtype='int32',
                                name='active_fake_extender_verb_agent')
    # TODO: reshaping constant input??
    tmp_reshaped_fake = Lambda(lambda x: K.tf.reshape(x, (fake_size, 1)))(const_fake_extender)

    inputs, output = single_sum_block(decrementing_input=flattened_decrementing_input,
                                      incrementing_input=flattened_incrementing_input,
                                      constant_input=tmp_reshaped_fake,
                                      roles=roles,
                                      filler_len=filler_len,
                                      dual_roles=dual_roles,
                                      max_depth=max_depth)

    return Model(
        inputs=[
            flattened_decrementing_input,
            flattened_incrementing_input,
            *inputs,
            const_fake_extender
        ],
        outputs=output)
