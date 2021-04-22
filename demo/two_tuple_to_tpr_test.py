import unittest

from core.model_2_tuple.core import Model2Tuple
from demo.two_tuple_to_tpr import aggregate_and_check


class Model2TupleToTPRTest(unittest.TestCase):

    def test_ideal_1(self):
        linguistic_scale_size = 5
        first_tuple = Model2Tuple(term_index=3, alpha=0, linguistic_scale_size=linguistic_scale_size)
        second_tuple = Model2Tuple(term_index=2, alpha=0, linguistic_scale_size=linguistic_scale_size)
        aggregate_and_check(first_tuple, second_tuple)

    def test_ideal_2(self):
        linguistic_scale_size = 5
        first_tuple = Model2Tuple(term_index=3, alpha=0.2, linguistic_scale_size=linguistic_scale_size)
        second_tuple = Model2Tuple(term_index=2, alpha=0, linguistic_scale_size=linguistic_scale_size)
        aggregate_and_check(first_tuple, second_tuple)

    def test_ideal_3(self):
        linguistic_scale_size = 5
        first_tuple = Model2Tuple(term_index=4, alpha=0.2, linguistic_scale_size=linguistic_scale_size)
        second_tuple = Model2Tuple(term_index=1, alpha=0, linguistic_scale_size=linguistic_scale_size)
        aggregate_and_check(first_tuple, second_tuple)
