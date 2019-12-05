def build_extraction_branch(model_input, roles, filler_len, max_depth, stop_level, role_extraction_order):
    shift_inputs = []
    current_input = model_input
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
    return shift_inputs, current_input

def extract_semantic_tree_from_passive_voice_branch():
    verb_extraction_inputs, verb_extraction_output = build_extraction_branch(model_input=flattened_tree_input,
                                                                   roles=roles,
                                                                   filler_len=filler_len,
                                                                   max_depth=max_depth,
                                                                   stop_level=stop_level,
                                                                   role_extraction_order=[1, 0, 1])
    agent_extraction_inputs, agent_extraction_output = build_extraction_branch(model_input=flattened_tree_input,
                                                                   roles=roles,
                                                                   filler_len=filler_len,
                                                                   max_depth=max_depth,
                                                                   stop_level=stop_level,
                                                                   role_extraction_order=[1, 1, 1])
    p_extraction_inputs, p_extraction_output = build_extraction_branch(model_input=flattened_tree_input,
                                                                   roles=roles,
                                                                   filler_len=filler_len,
                                                                   max_depth=max_depth,
                                                                   stop_level=stop_level,
                                                                   role_extraction_order=[0])

    agentxr0_pxr1_inputs, agentxr0_pxr1_output = build_join_branch(model_input=flattened_tree_input,
                                                                   roles=roles,
                                                                   filler_len=filler_len,
                                                                   max_depth=max_depth,
                                                                   stop_level=stop_level,
                                                                   role_extraction_order=[0],
                                                           inputs=[
                                                               agent_extraction_output,
                                                               p_extraction_output
                                                           ])

    semantic_tree_inputs, semantic_tree_output = build_join_branch(model_input=flattened_tree_input,
                                                                   roles=roles,
                                                                   filler_len=filler_len,
                                                                   max_depth=max_depth,
                                                                   stop_level=stop_level,
                                                                   role_extraction_order=[0],
                                                           inputs=[
                                                               verb_extraction_output,
                                                               agentxr0_pxr1_output
                                                           ])
    return semantic_tree_inputs, semantic_tree_output
