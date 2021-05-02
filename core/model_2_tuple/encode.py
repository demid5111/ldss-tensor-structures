import numpy as np

from core.active_passive_net.utils import elementary_join
from core.joiner.utils import generate_shapes
from core.joiner.vendor.network import build_tree_joiner_network
from .core import Model2Tuple
from .filler_factory import FillerFactory


def encode_model_2_tuple(model_2_tuple: Model2Tuple, encoder=None) -> np.array:
    roles = np.array([
        [1, 0, 0],  # r_i
        [0, 1, 0],  # r_alpha
        [0, 0, 1],  # r_w
    ])

    filler_index, filler_alpha, filler_weight = FillerFactory.from_model_2_tuple(model_2_tuple)

    MAX_TREE_DEPTH = 2
    SINGLE_ROLE_SHAPE = roles[0].shape
    SINGLE_FILLER_SHAPE = filler_index.shape

    fillers_shapes = generate_shapes(max_tree_depth=MAX_TREE_DEPTH,
                                     role_shape=SINGLE_ROLE_SHAPE,
                                     filler_shape=SINGLE_FILLER_SHAPE)

    if encoder is None:
        encoder = build_tree_joiner_network(roles=roles, fillers_shapes=fillers_shapes)

    fillers = np.array([filler_index, filler_alpha, filler_weight])
    model_2_tuple_encoded = elementary_join(joiner_network=encoder,
                                            input_structure_max_shape=fillers_shapes,
                                            basic_roles=roles,
                                            basic_fillers=fillers,
                                            subtrees=(
                                                filler_index,
                                                filler_alpha,
                                                filler_weight
                                            ))
    return model_2_tuple_encoded, encoder
