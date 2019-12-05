from core.active_passive_net.classifier.vendor.network import build_one_level_extraction_branch


def extract_semantic_tree_from_passive_voice_branch(input_layer, roles, filler_len, max_depth):
    stop_level_for_verb = 0
    verb_branch = build_one_level_extraction_branch(model_input=input_layer,
                                                    roles=roles,
                                                    filler_len=filler_len,
                                                    max_depth=max_depth,
                                                    stop_level=stop_level_for_verb,
                                                    role_extraction_order=[1, 0, 1])
    verb_extraction_const_inputs, verb_extraction_output = verb_branch

    stop_level_for_agent = 0
    agent_branch = build_one_level_extraction_branch(model_input=input_layer,
                                                     roles=roles,
                                                     filler_len=filler_len,
                                                     max_depth=max_depth,
                                                     stop_level=stop_level_for_agent,
                                                     role_extraction_order=[1, 1, 1])
    agent_extraction_const_inputs, agent_extraction_output = agent_branch

    stop_level_for_p = max_depth - 1
    p_branch = build_one_level_extraction_branch(model_input=input_layer,
                                                 roles=roles,
                                                 filler_len=filler_len,
                                                 max_depth=max_depth,
                                                 stop_level=stop_level_for_p,
                                                 role_extraction_order=[0])

    p_extraction_const_inputs, p_extraction_output = p_branch

    # TODO: define how to tackle extractions not till the bottom of structure
    _, agentxr0_pxr1_output = build_join_branch(model_input=flattened_tree_input,
                                                roles=roles,
                                                filler_len=filler_len,
                                                max_depth=max_depth,
                                                stop_level=stop_level,
                                                role_extraction_order=[0],
                                                inputs=[
                                                    agent_extraction_output,
                                                    p_extraction_output
                                                ])

    _, semantic_tree_output = build_join_branch(model_input=flattened_tree_input,
                                                roles=roles,
                                                filler_len=filler_len,
                                                max_depth=max_depth,
                                                stop_level=stop_level,
                                                role_extraction_order=[0],
                                                inputs=[
                                                    verb_extraction_output,
                                                    agentxr0_pxr1_output
                                                ])
    return [
               *verb_extraction_const_inputs,
               *agent_extraction_const_inputs,
               *p_extraction_const_inputs
           ], semantic_tree_output
