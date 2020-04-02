import numpy as np


def generate_shapes(max_tree_depth, role_shape, filler_shape):
    shapes = []
    for i in range(max_tree_depth):
        roles_shape_addon = [role_shape[0] for _ in range(i)]
        shapes.append(np.array([1, filler_shape[0], *roles_shape_addon]))
    return np.array(shapes)


def generate_input_placeholder(fillers_shapes):
    return [np.zeros(shape) for shape in fillers_shapes]


def extract_per_level_tensor_representation_after_shift(fillers_joined, max_tree_depth, role_shape, filler_shape):
    levels = []
    slicing_index = 0

    expected_shapes = generate_shapes(max_tree_depth=max_tree_depth + 1,
                                      role_shape=role_shape,
                                      filler_shape=filler_shape)
    for shape in expected_shapes[1:]:
        max_index = np.prod(shape)
        product = fillers_joined[slicing_index:slicing_index + max_index].reshape(shape)
        slicing_index = slicing_index + max_index
        levels.append(product)
    return levels


def reshape_to_satisfy_max_depth_after_shift(tensor_representation, max_tree_depth, role_shape, filler_shape):
    expected_shapes = generate_shapes(max_tree_depth=max_tree_depth,
                                      role_shape=role_shape,
                                      filler_shape=filler_shape)

    res_representation = [None for _ in range(max_tree_depth)]
    existing_levels = set()
    for level_representation in tensor_representation:
        for j, expected_shape in enumerate(expected_shapes):
            if np.array_equal(level_representation.shape, expected_shape):
                existing_levels.add(j)
                res_representation[j] = level_representation
                break

    for i, el in enumerate(res_representation):
        if el is not None:
            continue
        res_representation[i] = np.zeros(expected_shapes[i])
    return res_representation
