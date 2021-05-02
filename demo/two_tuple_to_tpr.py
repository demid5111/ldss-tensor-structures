import tensorflow as tf

from core.model_2_tuple.aggregation import aggregate_and_check
from core.model_2_tuple.core import Model2Tuple


def main():
    print('Converting 2-tuple to TPR')
    tf.compat.v1.disable_eager_execution()
    linguistic_scale_size = 5
    first_tuple = Model2Tuple(term_index=4, alpha=0.2, linguistic_scale_size=linguistic_scale_size)
    second_tuple = Model2Tuple(term_index=3, alpha=-0.1, linguistic_scale_size=linguistic_scale_size)

    aggregate_and_check(first_tuple, second_tuple)
    print('Converting 2-tuple to TPR and back works with no information loss!')


if __name__ == '__main__':
    main()
