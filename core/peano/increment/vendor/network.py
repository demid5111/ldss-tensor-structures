from keras import Model, Input
from keras.layers import Concatenate

from core.active_passive_net.active_extractor.vendor.network import custom_cropping_layer, custom_constant_layer
from core.active_passive_net.classifier.vendor.network import build_one_level_extraction_branch
from core.active_passive_net.passive_extractor.vendor.network import build_join_branch
from core.unshifter.vendor.network import unshift_matrix
from demo.active_passive_net import elementary_join, get_filler_by


def number_to_tree(target_number, joiner_network, maximum_shapes, fillers, roles, order_case_active):
    one = get_filler_by(name='A', order=order_case_active, fillers=fillers)
    for i in range(1):
        # one is a result of two joins
        # 1. join of filler and role_1 a.k.a zero representation
        # 2. join of step 1 and role_1 a.k.a zero representation
        one = elementary_join(joiner_network=joiner_network,
                              input_structure_max_shape=maximum_shapes,
                              basic_roles=roles,
                              basic_fillers=fillers,
                              subtrees=(
                                  None,
                                  one
                              ))

    number = one
    for i in range(target_number - 1):
        # easy as 2 is just one join of (one+one)
        # easy as 3 is just two joins: (one+one)+one
        number = elementary_join(joiner_network=joiner_network,
                                 input_structure_max_shape=maximum_shapes,
                                 basic_roles=roles,
                                 basic_fillers=fillers,
                                 subtrees=(
                                     number,
                                     one
                                 ))
    return number


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


def single_sum_block(decrementing_input, incrementing_input, constant_input_one, constant_input_filler, roles,
                     filler_len, dual_roles, max_depth, increment_value):
    block_id = 1

    const_inputs, one_tensor_output = build_extract_branch(
        input_layer=decrementing_input,
        extract_role=dual_roles[1],
        filler_len=filler_len,
        max_depth=max_depth - 1,
        branch_id=block_id
    )

    concatenate_one = Concatenate(axis=0)([one_tensor_output, constant_input_one])

    next_number_const_inputs, next_number_output = build_join_branch(
        roles=roles,
        filler_len=filler_len,
        max_depth=max_depth,
        inputs=[
            incrementing_input,
            increment_value
        ],
        prefix='cons_'.format(block_id)
    )
    concatenate_sum = Concatenate(axis=0)([constant_input_filler, next_number_output])

    target_num_elements, flattened_num_elements = unshift_matrix(dual_roles[1], filler_len, max_depth).shape
    cropped_number = custom_cropping_layer(
        input_layer=concatenate_sum,
        crop_from_beginning=0,
        crop_from_end=flattened_num_elements + filler_len - target_num_elements,
        input_tensor_length=flattened_num_elements + filler_len,
        final_tensor_length=target_num_elements
    )
    return [
               *const_inputs,
               *next_number_const_inputs
           ], cropped_number


def build_increment_network(roles, fillers, dual_roles, max_depth, increment_value):
    filler_len = fillers[0].shape[0]

    input_num_elements, flattened_tree_num_elements = unshift_matrix(roles[0], filler_len, max_depth - 1).shape
    shape = (flattened_tree_num_elements + filler_len, 1)
    flattened_decrementing_input = Input(shape=(*shape,), batch_shape=(*shape,))
    flattened_incrementing_input = Input(shape=(*shape,), batch_shape=(*shape,))

    # _, flattened_tree_num_elements_extender = unshift_matrix(roles[0], filler_len, max_depth-1).shape
    # later we have to join two subtrees of different depth. for that we have to
    # make filler of verb of the same depth - make fake constant layer
    tmp_reshaped_fake, const_one = custom_constant_layer(const_size=input_num_elements + filler_len, name='const_one')

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
