import unittest
import tensorflow as tf

from core.model_2_tuple import aggregate_and_check
from core.model_2_tuple import Model2Tuple


class Model2TupleToTPRTest(unittest.TestCase):

    def test_ideal_1(self):
        tf.compat.v1.disable_eager_execution()
        linguistic_scale_size = 5
        first_tuple = Model2Tuple(term_index=3, alpha=0, linguistic_scale_size=linguistic_scale_size)
        second_tuple = Model2Tuple(term_index=2, alpha=0, linguistic_scale_size=linguistic_scale_size)
        aggregate_and_check(first_tuple, second_tuple)

    def test_ideal_2(self):
        tf.compat.v1.disable_eager_execution()
        linguistic_scale_size = 5
        first_tuple = Model2Tuple(term_index=3, alpha=0.2, linguistic_scale_size=linguistic_scale_size)
        second_tuple = Model2Tuple(term_index=2, alpha=0, linguistic_scale_size=linguistic_scale_size)
        aggregate_and_check(first_tuple, second_tuple)

    def test_ideal_3(self):
        tf.compat.v1.disable_eager_execution()
        linguistic_scale_size = 5
        first_tuple = Model2Tuple(term_index=4, alpha=0.2, linguistic_scale_size=linguistic_scale_size)
        second_tuple = Model2Tuple(term_index=1, alpha=0, linguistic_scale_size=linguistic_scale_size)
        aggregate_and_check(first_tuple, second_tuple)

    def test_ideal_negative_weights_1(self):
        tf.compat.v1.disable_eager_execution()
        linguistic_scale_size = 5
        first_tuple = Model2Tuple(term_index=4, alpha=0.2, linguistic_scale_size=linguistic_scale_size)
        second_tuple = Model2Tuple(term_index=3, alpha=-0.1, linguistic_scale_size=linguistic_scale_size)
        aggregate_and_check(first_tuple, second_tuple)
