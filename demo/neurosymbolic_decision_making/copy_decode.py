import numpy as np

from core.model_2_tuple import Model2Tuple, FillerFactory
from core.utils import flattenize_per_tensor_representation
from demo.neurosymbolic_decision_making.copy_decoder import build_decode_model_2_tuple_network


def decode_model_2_tuple_tpr(mta_result_encoded: np.array, decoder=None, model_2_tuple_has_weights=True):
    if model_2_tuple_has_weights:
        roles = np.array([
            [1, 0, 0],  # r_i
            [0, 1, 0],  # r_alpha
            [0, 0, 1],  # r_w
        ])
        dual_roles = np.linalg.inv(roles)
    else:
        roles = np.array([
            [1, 0],  # r_i
            [0, 1],  # r_alpha
        ])
        dual_roles = np.linalg.inv(roles)

    weight = 0.0 if model_2_tuple_has_weights else None
    model_2_tuple = Model2Tuple(term_index=0, alpha=.0, linguistic_scale_size=5, weight=weight)

    filler_len = FillerFactory.get_filler_size(model_2_tuple)
    MAX_TREE_DEPTH = 2

    if decoder is None:
        decoder = build_decode_model_2_tuple_network(filler_len=filler_len, dual_roles=dual_roles,
                                                     max_depth=MAX_TREE_DEPTH,
                                                     model_2_tuple_has_weights=model_2_tuple_has_weights)

    if not hasattr(mta_result_encoded, 'shape') or len(mta_result_encoded.shape) > 1:
        flattened_model_2_tuple_tpr = flattenize_per_tensor_representation(mta_result_encoded)
    else:
        flattened_model_2_tuple_tpr = mta_result_encoded
    flattened_model_2_tuple_tpr = flattened_model_2_tuple_tpr.reshape((1, *flattened_model_2_tuple_tpr.shape, 1))

    if model_2_tuple_has_weights:
        filler_index, filler_alpha, filler_weight = decoder.predict_on_batch([
            flattened_model_2_tuple_tpr
        ])

        term_index, alpha, weight = FillerFactory.decode_tpr(filler_index, filler_alpha, filler_weight)
    else:
        filler_index, filler_alpha = decoder.predict_on_batch([
            flattened_model_2_tuple_tpr
        ])

        term_index, alpha, weight = FillerFactory.decode_tpr(filler_index, filler_alpha)

    return Model2Tuple(term_index=term_index, alpha=alpha, weight=weight), decoder
