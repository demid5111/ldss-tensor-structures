import unittest

from core.model_2_tuple.core import Model2Tuple
from demo.two_tuple_to_tpr import encode_model_2_tuple, decode_model_2_tuple_tpr


class Model2TupleToTPRTest(unittest.TestCase):
    def aggregate_and_check(self, model_2_tuple_a, model_2_tuple_b):
        self.assertEqual(model_2_tuple_a.linguistic_scale_size, model_2_tuple_b.linguistic_scale_size)

        aggregation = model_2_tuple_a.to_number() * model_2_tuple_a.weight + \
                      model_2_tuple_b.to_number() * model_2_tuple_b.weight
        mta_result = Model2Tuple.from_number(beta=aggregation,
                                             linguistic_scale_size=model_2_tuple_a.linguistic_scale_size)

        mta_result_encoded = encode_model_2_tuple(mta_result)
        decoded_2_tuple = decode_model_2_tuple_tpr(mta_result_encoded)

        self.assertEqual(mta_result, decoded_2_tuple, 'Encoding is working with no information loss')

    def test_ideal_1(self):
        linguistic_scale_size = 5
        first_tuple = Model2Tuple(term_index=3, alpha=0, linguistic_scale_size=linguistic_scale_size)
        second_tuple = Model2Tuple(term_index=2, alpha=0, linguistic_scale_size=linguistic_scale_size)
        self.aggregate_and_check(first_tuple, second_tuple)

    def test_ideal_2(self):
        linguistic_scale_size = 5
        first_tuple = Model2Tuple(term_index=3, alpha=0.2, linguistic_scale_size=linguistic_scale_size)
        second_tuple = Model2Tuple(term_index=2, alpha=0, linguistic_scale_size=linguistic_scale_size)
        self.aggregate_and_check(first_tuple, second_tuple)

    def test_ideal_3(self):
        linguistic_scale_size = 5
        first_tuple = Model2Tuple(term_index=4, alpha=0.2, linguistic_scale_size=linguistic_scale_size)
        second_tuple = Model2Tuple(term_index=1, alpha=0, linguistic_scale_size=linguistic_scale_size)
        self.aggregate_and_check(first_tuple, second_tuple)
