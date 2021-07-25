from functools import reduce
from typing import List

from .core import Model2Tuple
from .decode import decode_model_2_tuple_tpr
from .encode import encode_model_2_tuple


def aggregate_model_tuples(tuples: List[Model2Tuple], linguistic_scale_size):
    avg_weight = 1 / len(tuples)
    weights = map(lambda y: y.weight or avg_weight, tuples)
    mta_result_number = reduce(lambda acc, y: acc + y[0].to_number() * y[1], zip(tuples, weights), 0)

    has_weights = all(map(lambda y: y.weight, tuples))
    final_weight = 0.0 if has_weights else None
    return Model2Tuple.from_number(beta=mta_result_number, linguistic_scale_size=linguistic_scale_size,
                                   weight=final_weight)


def aggregate_and_check(model_2_tuple_a, model_2_tuple_b):
    if model_2_tuple_a.linguistic_scale_size != model_2_tuple_b.linguistic_scale_size:
        raise ValueError('2-tuple elements should be from the single scale!')

    mta_result = aggregate_model_tuples([model_2_tuple_a, model_2_tuple_b], model_2_tuple_a.linguistic_scale_size)

    mta_result_encoded, _ = encode_model_2_tuple(mta_result)

    has_weights = mta_result.weight is not None
    decoded_2_tuple, _ = decode_model_2_tuple_tpr(mta_result_encoded, model_2_tuple_has_weights=has_weights)

    if mta_result != decoded_2_tuple:
        raise ValueError('Encoding is working with information loss!')
