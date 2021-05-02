from functools import reduce
from typing import List

from core.model_2_tuple.core import Model2Tuple
from core.model_2_tuple.decode import decode_model_2_tuple_tpr
from core.model_2_tuple.encode import encode_model_2_tuple


def aggregate_model_tuples(tuples: List[Model2Tuple], linguistic_scale_size):
    mta_result_number = reduce(lambda acc, y: acc + y.to_number() * y.weight, tuples, 0)
    return Model2Tuple.from_number(beta=mta_result_number, linguistic_scale_size=linguistic_scale_size)


def aggregate_and_check(model_2_tuple_a, model_2_tuple_b):
    if model_2_tuple_a.linguistic_scale_size != model_2_tuple_b.linguistic_scale_size:
        raise ValueError('2-tuple elements should be from the single scale!')

    mta_result = aggregate_model_tuples([model_2_tuple_a, model_2_tuple_b], model_2_tuple_a.linguistic_scale_size)

    mta_result_encoded = encode_model_2_tuple(mta_result)
    decoded_2_tuple = decode_model_2_tuple_tpr(mta_result_encoded)

    if mta_result != decoded_2_tuple:
        raise ValueError('Encoding is working with information loss!')
