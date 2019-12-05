def extract_semantic_tree_from_active_voice_branch():
    verb_extraction_inputs, verb_extraction_output = build_extraction_branch(model_input=flattened_tree_input,
                                                                   roles=roles,
                                                                   filler_len=filler_len,
                                                                   max_depth=max_depth,
                                                                   stop_level=stop_level,
                                                                   role_extraction_order=[1, 0])
    agent_extraction_inputs, agent_extraction_output = build_extraction_branch(model_input=flattened_tree_input,
                                                                   roles=roles,
                                                                   filler_len=filler_len,
                                                                   max_depth=max_depth,
                                                                   stop_level=stop_level,
                                                                   role_extraction_order=[0])
    p_extraction_inputs, p_extraction_output = build_extraction_branch(model_input=flattened_tree_input,
                                                                   roles=roles,
                                                                   filler_len=filler_len,
                                                                   max_depth=max_depth,
                                                                   stop_level=stop_level,
                                                                   role_extraction_order=[1, 1])

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
