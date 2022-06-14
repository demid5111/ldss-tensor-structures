import numpy as np
import tensorflow as tf

from core.active_passive_net.classifier.vendor.network import build_one_level_extraction_branch
from core.active_passive_net.passive_extractor.vendor.network import build_join_branch
from core.unshifter.vendor.network import unshift_matrix
from core.utils import keras_constant_layer


def crop_tensor(layer, role, filler_len, stop_level):
    _, flattened_num_elements = unshift_matrix(role, filler_len, stop_level).shape
    return custom_cropping_layer(input_layer=layer,
                                 crop_from_beginning=0,
                                 crop_from_end=flattened_num_elements,
                                 input_tensor_length=flattened_num_elements + filler_len,
                                 final_tensor_length=filler_len)


def custom_cropping_layer(input_layer, crop_from_beginning, crop_from_end, input_tensor_length, final_tensor_length):
    reshape_for_crop = tf.keras.layers.Lambda(lambda x: tf.keras.backend.reshape(x, (1, input_tensor_length, 1)))(
        input_layer)
    clip_first_level = tf.keras.layers.Cropping1D(cropping=(crop_from_beginning, crop_from_end))(reshape_for_crop)
    return tf.keras.layers.Lambda(lambda x: tf.keras.backend.reshape(x, (1, final_tensor_length, 1)))(clip_first_level)


def custom_constant_layer(const_size, name, np_constant=None):
    if np_constant is None:
        np_constant = np.zeros((const_size, 1))
    else:
        np_constant = np.reshape(np_constant, (*np_constant.shape, 1))

    const_fake_extender = keras_constant_layer(np_constant, name=name)
    # tf_constant = tf.keras.backend.constant(np_constant, dtype='float32')
    # const_fake_extender = Input(tensor=tf_constant, shape=np_constant.shape, dtype='float32', name=name)
    # TODO: reshaping constant input??
    return tf.keras.layers.Lambda(lambda x: tf.keras.backend.reshape(x, const_fake_extender.shape))(
        const_fake_extender), const_fake_extender


def extract_semantic_tree_from_active_voice_branch(input_layer, roles, dual_roles, filler_len, max_depth):
    stop_level_for_verb = max_depth - 2
    verb_branch = build_one_level_extraction_branch(model_input=input_layer,
                                                    roles=dual_roles,
                                                    filler_len=filler_len,
                                                    max_depth=max_depth,
                                                    stop_level=stop_level_for_verb,
                                                    role_extraction_order=[1, 0],
                                                    prefix='active_verb_extract')
    verb_raw_output, _ = verb_branch
    verb_extraction_output = crop_tensor(layer=verb_raw_output,
                                         role=roles[0],
                                         filler_len=filler_len,
                                         stop_level=stop_level_for_verb)

    stop_level_for_agent = max_depth - 1
    agent_branch = build_one_level_extraction_branch(model_input=input_layer,
                                                     roles=dual_roles,
                                                     filler_len=filler_len,
                                                     max_depth=max_depth,
                                                     stop_level=stop_level_for_agent,
                                                     role_extraction_order=[0],
                                                     prefix='active_agent_extract')
    agent_raw_output, _ = agent_branch
    agent_extraction_output = crop_tensor(layer=agent_raw_output,
                                          role=roles[0],
                                          filler_len=filler_len,
                                          stop_level=stop_level_for_agent)

    stop_level_for_p = max_depth - 2
    p_branch = build_one_level_extraction_branch(model_input=input_layer,
                                                 roles=dual_roles,
                                                 filler_len=filler_len,
                                                 max_depth=max_depth,
                                                 stop_level=stop_level_for_p,
                                                 role_extraction_order=[1, 1],
                                                 prefix='active_p_extract')

    p_raw_output, current_num_elements = p_branch
    p_extraction_output = crop_tensor(layer=p_raw_output,
                                      role=roles[0],
                                      filler_len=filler_len,
                                      stop_level=stop_level_for_p)

    # TODO: define how to tackle extractions not till the bottom of structure
    # given that we have all fillers maximum joining depth is equal to 1
    agentxr0_pxr1_output = build_join_branch(roles=roles,
                                                                         filler_len=filler_len,
                                                                         max_depth=1,
                                                                         inputs=[
                                                                             agent_extraction_output,
                                                                             p_extraction_output
                                                                         ],
                                                                         prefix='active_join(agent,p)'
                                                                         )

    # later we have to join two subtrees of different depth. for that we have to
    # make filler of verb of the same depth - make fake constant layer
    tmp_reshaped_fake, const_input = custom_constant_layer(const_size=filler_len,
                                                           name='active_fake_extender_verb_agent')
    concatenate_verb = tf.keras.layers.Concatenate(axis=1)(
        [verb_extraction_output, tmp_reshaped_fake, tmp_reshaped_fake])
    # TODO: why is there a constant 3?
    reshaped_verb = tf.keras.layers.Lambda(lambda x: tf.keras.backend.reshape(x, (1, filler_len * 3, 1)))(
        concatenate_verb)

    # TODO: reshape by 2, why is there a constant 2?
    tmp_reshaped_agentxr0_pxr1 = tf.keras.layers.Lambda(lambda x: tf.keras.backend.reshape(x, (1, filler_len * 2, 1)))(
        agentxr0_pxr1_output)
    concatenate_agentxr0_pxr1 = tf.keras.layers.Concatenate(axis=1)([tmp_reshaped_fake, tmp_reshaped_agentxr0_pxr1])
    # TODO: why is there a constant 3?
    reshaped_agentxr0_pxr1 = tf.keras.layers.Lambda(lambda x: tf.keras.backend.reshape(x, (1, filler_len * 3, 1)))(
        concatenate_agentxr0_pxr1)

    semantic_tree_output = build_join_branch(roles=roles,
                                                                         filler_len=filler_len,
                                                                         max_depth=2,
                                                                         inputs=[
                                                                             reshaped_verb,
                                                                             reshaped_agentxr0_pxr1
                                                                         ],
                                                                         prefix='active_join(verb, join(agent,p))')
    return [
               const_input
           ], semantic_tree_output
