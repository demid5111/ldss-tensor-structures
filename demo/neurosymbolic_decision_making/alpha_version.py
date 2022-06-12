import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from core.joiner.utils import generate_shapes
from core.model_2_tuple import Model2Tuple, aggregate_model_tuples, encode_model_2_tuple, FillerFactory
from demo.neurosymbolic_decision_making.encoder import build_tree_joiner_network


def generate_assessments(num_assessments, linguistic_scale_size):
    assessments = []
    for _ in range(num_assessments):
        random_index = random.randint(0, 4)
        random_alpha = random.uniform(-0.4, 0.4)
        assessment = Model2Tuple(term_index=random_index,
                                 alpha=random_alpha,
                                 linguistic_scale_size=linguistic_scale_size)
        assessments.append(assessment)
    return assessments


def symbolic_aggregation(assessments, linguistic_scale_size):
    print('Step 2. Process assessments (translate to the distributed form if needed)')
    print('\tnot needed')

    print('Step 3. Aggregate assessments (symbolic-/sub-symbolically)')
    print('\tsymbolically')
    res_assessment = aggregate_model_tuples(assessments, linguistic_scale_size=linguistic_scale_size)

    print('Step 4. Process assessments (translate from the distributed form if needed)')
    print('\tnot needed')

    print('Step 5. Communicating result')
    print(f'\tResulting assessment: {res_assessment}')


def subsymbolic_aggregation(assessments):
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.disable_eager_execution()

    print('Step 2. Process assessments (translate to the distributed form if needed)')
    encoder_dir = Path(__file__).parent / 'models'
    encoder_path = encoder_dir / 'encoder.savedmodel'
    if not encoder_path.exists():
        print('\tEncoder model does not exist, need to save it first')
        _, encoder = encode_model_2_tuple(assessments[0], encoder=None)
        encoder.save(str(encoder_path))

    encoder = tf.keras.models.load_model(str(encoder_path))
    encoded_assessments = [encode_model_2_tuple(i, encoder=encoder)[0] for i in assessments]
    print(encoded_assessments)


def main():
    is_symbolic = True
    is_subsymbolic = True
    num_assessments = 2
    linguistic_scale_size = 5

    print('Step 1. Obtain assessments (read or generate)')
    print('\tGenerating assessments...')
    assessments = generate_assessments(num_assessments, linguistic_scale_size)
    print(f'\tAssessments: {", ".join([str(i) for i in assessments])}')

    # if is_symbolic:
    #     symbolic_aggregation(assessments, linguistic_scale_size)
    if is_subsymbolic:
        subsymbolic_aggregation(assessments)


if __name__ == '__main__':
    main()
