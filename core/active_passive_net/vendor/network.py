from keras.models import Model
from keras.layers import Lambda, Add

from core.active_passive_net.active_extractor.vendor.network import extract_semantic_tree_from_active_voice_branch
from core.active_passive_net.classifier.vendor.network import build_classification_branch
from core.active_passive_net.passive_extractor.vendor.network import extract_semantic_tree_from_passive_voice_branch


def build_active_passive_network(roles, dual_roles, fillers, tree_shape):
    filler_len = fillers[0].shape[0]
    max_depth = len(tree_shape) - 1

    branch = build_classification_branch(roles=dual_roles,
                                         fillers=fillers,
                                         tree_shape=tree_shape,
                                         role_extraction_order=[1,
                                                                0,
                                                                0],
                                         stop_level=0)
    classification_const_inputs, apnet_variable_input, classification_output = branch

    scalar_mul = Lambda(lambda tensors: tensors[0] * tensors[1])([apnet_variable_input, classification_output])
    passive_branch_const_inputs, passive_branch_output = extract_semantic_tree_from_passive_voice_branch(
        input_layer=scalar_mul,
        roles=dual_roles,
        filler_len=filler_len,
        max_depth=max_depth)
    active_branch_const_inputs, active_branch_output = extract_semantic_tree_from_active_voice_branch(
        input_layer=scalar_mul)

    sum_branches = Add()([passive_branch_output, active_branch_output])
    return Model(
        inputs=[
            *classification_const_inputs,
            *passive_branch_const_inputs,
            *active_branch_const_inputs,
            apnet_variable_input,
        ],
        outputs=sum_branches)
