import keras.backend as K
import numpy as np
from keras import Input
from keras.layers import Lambda, Cropping1D, Add, Concatenate

from core.active_passive_net.classifier.vendor.network import build_one_level_extraction_branch
from core.joiner.vendor.network import constant_input, mat_mul, shift_matrix
from core.unshifter.vendor.network import unshift_matrix


def create_shift_matrix_as_input(role, role_index, filler_len, max_depth, prefix):
    left_shift_input_name = '{}constant_input_(cons{})'.format(prefix + '_' if prefix else '', role_index)
    return constant_input(role, filler_len, max_depth, left_shift_input_name, shift_matrix)


# TODO: refactor and move to the joiner network
def build_join_branch(roles, filler_len, max_depth, inputs, prefix='', left_shift_input=None, right_shift_input=None):
    if left_shift_input is None:
        left_shift_input = create_shift_matrix_as_input(roles[0], 0, filler_len, max_depth, prefix)

    if right_shift_input is None:
        right_shift_input = create_shift_matrix_as_input(roles[1], 1, filler_len, max_depth, prefix)

    left_matmul_layer = Lambda(mat_mul)([
        left_shift_input,
        inputs[0]
    ])

    right_matmul_layer = Lambda(mat_mul)([
        right_shift_input,
        inputs[1]
    ])

    sum_layer = Add()([
        left_matmul_layer,
        right_matmul_layer
    ])

    return (
               left_shift_input,
               right_shift_input
           ), sum_layer


def extract_semantic_tree_from_passive_voice_branch(input_layer, roles, dual_roles, filler_len, max_depth):
    stop_level_for_verb = 0
    verb_branch = build_one_level_extraction_branch(model_input=input_layer,
                                                    roles=dual_roles,
                                                    filler_len=filler_len,
                                                    max_depth=max_depth,
                                                    stop_level=stop_level_for_verb,
                                                    role_extraction_order=[1, 0, 1],
                                                    prefix='passive_verb_extract')
    verb_extraction_const_inputs, verb_extraction_output, _ = verb_branch

    stop_level_for_agent = 0
    agent_branch = build_one_level_extraction_branch(model_input=input_layer,
                                                     roles=dual_roles,
                                                     filler_len=filler_len,
                                                     max_depth=max_depth,
                                                     stop_level=stop_level_for_agent,
                                                     role_extraction_order=[1, 1, 1],
                                                     prefix='passive_agent_extract')
    agent_extraction_const_inputs, agent_extraction_output, _ = agent_branch

    stop_level_for_p = max_depth - 1
    p_branch = build_one_level_extraction_branch(model_input=input_layer,
                                                 roles=dual_roles,
                                                 filler_len=filler_len,
                                                 max_depth=max_depth,
                                                 stop_level=stop_level_for_p,
                                                 role_extraction_order=[0],
                                                 prefix='passive_p_extract')

    p_extraction_const_inputs, p_raw_output, current_num_elements = p_branch
    _, flattened_num_elements = unshift_matrix(roles[0], filler_len, stop_level_for_p).shape
    # TODO: insert cropping here
    reshape_for_crop = Lambda(lambda x: K.tf.reshape(x, (1, flattened_num_elements + filler_len, 1)))(p_raw_output)
    clip_first_level = Cropping1D(cropping=(0, flattened_num_elements))(reshape_for_crop)
    p_extraction_output = Lambda(lambda x: K.tf.reshape(x, (filler_len, 1)))(clip_first_level)

    # TODO: define how to tackle extractions not till the bottom of structure
    # given that we have all fillers maximum joining depth is equal to 1
    agentxr0_pxr1_const_inputs, agentxr0_pxr1_output = build_join_branch(roles=roles,
                                                                         filler_len=filler_len,
                                                                         max_depth=1,
                                                                         inputs=[
                                                                             agent_extraction_output,
                                                                             p_extraction_output
                                                                         ],
                                                                         prefix='passive_join(agent,p)'
                                                                         )

    # later we have to join two subtrees of different depth. for that we have to
    # make filler of verb of the same depth - make fake constant layer
    np_constant = np.zeros((filler_len, 1))
    tf_constant = K.constant(np_constant)
    const_fake_extender = Input(tensor=tf_constant,
                                batch_shape=np_constant.shape,
                                shape=np_constant.shape,
                                dtype='int32',
                                name='passive_fake_extender_verb_agent')
    concatenate_verb = Concatenate(axis=0)([verb_extraction_output, const_fake_extender, const_fake_extender])
    # TODO: why is there a constant 3?
    reshaped_verb = Lambda(lambda x: K.tf.reshape(x, (filler_len * 3, 1)))(concatenate_verb)

    # TODO: reshape by 2, why is there a constant 2?
    tmp_reshaped_agentxr0_pxr1 = Lambda(lambda x: K.tf.reshape(x, (filler_len * 2, 1)))(agentxr0_pxr1_output)
    # TODO: reshaping constant input??
    tmp_reshaped_fake = Lambda(lambda x: K.tf.reshape(x, (filler_len, 1)))(const_fake_extender)
    concatenate_agentxr0_pxr1 = Concatenate(axis=0)([tmp_reshaped_fake, tmp_reshaped_agentxr0_pxr1])
    # TODO: why is there a constant 3?
    reshaped_agentxr0_pxr1 = Lambda(lambda x: K.tf.reshape(x, (filler_len * 3, 1)))(concatenate_agentxr0_pxr1)

    semantic_tree_const_inputs, semantic_tree_output = build_join_branch(roles=roles,
                                                                         filler_len=filler_len,
                                                                         max_depth=2,
                                                                         inputs=[
                                                                             reshaped_verb,
                                                                             reshaped_agentxr0_pxr1
                                                                         ],
                                                                         prefix='passive_join(verb, join(agent,p))')
    return [
               *verb_extraction_const_inputs,
               *agent_extraction_const_inputs,
               *p_extraction_const_inputs,
               *agentxr0_pxr1_const_inputs,
               *semantic_tree_const_inputs,
               const_fake_extender
           ], semantic_tree_output
