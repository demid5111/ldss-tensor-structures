import numpy as np

from .core import Model2Tuple
from .decoder.vendor.network import build_decode_model_2_tuple_network
from .filler_factory import FillerFactory
from core.utils import flattenize_per_tensor_representation


def decode_model_2_tuple_tpr(mta_result_encoded: np.array, decoder=None):
    roles = np.array([
        [1, 0, 0],  # r_i
        [0, 1, 0],  # r_alpha
        [0, 0, 1],  # r_w
    ])
    dual_roles = np.linalg.inv(roles)
    filler_len = FillerFactory.get_filler_size()
    MAX_TREE_DEPTH = 2

    if decoder is None:
        decoder = build_decode_model_2_tuple_network(filler_len=filler_len, dual_roles=dual_roles,
                                                       max_depth=MAX_TREE_DEPTH)

    if not hasattr(mta_result_encoded, 'shape') or len(mta_result_encoded.shape) > 1:
        flattened_model_2_tuple_tpr = flattenize_per_tensor_representation(mta_result_encoded)
    else:
        flattened_model_2_tuple_tpr = mta_result_encoded
    flattened_model_2_tuple_tpr = flattened_model_2_tuple_tpr.reshape((1, *flattened_model_2_tuple_tpr.shape, 1))
    filler_index, filler_alpha, filler_weight = decoder.predict_on_batch([
        flattened_model_2_tuple_tpr
    ])

    term_index, alpha, weight = FillerFactory.decode_tpr(filler_index, filler_alpha, filler_weight)

    return Model2Tuple(term_index=term_index, alpha=alpha, weight=weight)
