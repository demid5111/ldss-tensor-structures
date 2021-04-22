import numpy as np

from core.joiner.utils import generate_input_placeholder, extract_per_level_tensor_representation_after_shift, \
    reshape_to_satisfy_max_depth_after_shift


def prepare_input(subtree, max_shape):
    if subtree is None:
        # there is no subtree for this role, therefore just generate the placeholder
        return generate_input_placeholder(max_shape)
    subtree_shapes = np.array(tuple(np.array(i.shape) for i in subtree))
    is_filler_subtree = hasattr(subtree, 'shape') and len(subtree.shape) == 1
    if not is_filler_subtree and \
            len(subtree_shapes) == len(max_shape) and \
            np.all([np.all(np.equal(subtree_shapes[i], max_shape[i])) for i, _ in enumerate(subtree_shapes)]):
        # TODO: need to understand why `extract_per_level_tensor_representation` returns a list
        # the subtree is already of a needed shape, just keep it unchanged
        return subtree

    if is_filler_subtree:
        # TODO: need to understand why `extract_per_level_tensor_representation` returns a list
        # subtree is a simple filler
        placeholder = generate_input_placeholder(max_shape)
        placeholder[0] = subtree.reshape(1, *subtree.shape)
        return placeholder

    raise NotImplementedError('This subtree cannot be prepared for join')


def elementary_join(joiner_network, input_structure_max_shape, basic_roles, basic_fillers, subtrees):
    input_tensors = map(lambda s: prepare_input(s, input_structure_max_shape), subtrees)

    fillers_joined, *not_needed = joiner_network.predict_on_batch([i for p in input_tensors for i in p])

    single_role_shape = basic_roles[0].shape
    single_filler_shape = basic_fillers[0].shape
    max_depth = input_structure_max_shape.shape[0]
    tensor_repr = extract_per_level_tensor_representation_after_shift(fillers_joined,
                                                                      max_tree_depth=max_depth,
                                                                      role_shape=single_role_shape,
                                                                      filler_shape=single_filler_shape)

    return reshape_to_satisfy_max_depth_after_shift(tensor_repr,
                                                    max_depth,
                                                    single_role_shape,
                                                    single_filler_shape)
