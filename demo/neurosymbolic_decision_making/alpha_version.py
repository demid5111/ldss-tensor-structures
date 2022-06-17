import random
import time
from pathlib import Path
from typing import Optional
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

tf.get_logger().setLevel('ERROR')
from tap import Tap

from core.model_2_tuple import Model2Tuple, aggregate_model_tuples, encode_model_2_tuple, decode_model_2_tuple_tpr

from demo.neurosymbolic_decision_making.copied_from_ldss_aggregator.generator import pack_with_full_mta_encoding, \
    single_tpr_len
from demo.neurosymbolic_decision_making.copied_from_ldss_aggregator.infer_mta import _generate_data, get_infer_routine
from demo.neurosymbolic_decision_making.copied_from_ldss_aggregator.task import MTATask
from demo.neurosymbolic_decision_making.copied_from_ldss_benchmark.schemas.task_scheme import \
    AlternativeAssessmentDescription, AlternativeAssessmentForSingleCriteriaDescription, ScalesDescription
from demo.neurosymbolic_decision_making.copied_from_ldss_benchmark.task_model import TaskModelFactory, TaskModel


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


def load_assessments(task_description_path: Path, num_assessments: int, linguistic_scale_size: int):
    task: TaskModel = TaskModelFactory().from_json(task_description_path)

    linguistic_scale: Optional[ScalesDescription] = None
    for scale in task._dto.scales:
        scale: ScalesDescription

        if len(scale.labels) == linguistic_scale_size and scale.values is None:
            linguistic_scale = scale
            break

    all_matching_assessments = []
    for expert_id, expert_assessments in task._dto.estimations.items():
        for alternative_assessment in expert_assessments:
            alternative_assessment: AlternativeAssessmentDescription

            for assessment in alternative_assessment.criteria2Estimation:
                assessment: AlternativeAssessmentForSingleCriteriaDescription

                if assessment.scaleID != linguistic_scale.scaleID:
                    continue
                label = assessment.estimation[0]
                position = linguistic_scale.labels.index(label)
                all_matching_assessments.append(Model2Tuple(term_index=position,
                                                            alpha=0.,
                                                            linguistic_scale_size=linguistic_scale_size))

    return all_matching_assessments[:num_assessments]


def load_encoder_decoder_models(encoder_path, decoder_path, has_weights=False):
    assert has_weights is False, 'weighted TPR encoding is not currently supported'

    linguistic_scale_size = 5
    fake_2_tuple = Model2Tuple(term_index=4, alpha=0.2, linguistic_scale_size=linguistic_scale_size)
    if not encoder_path.exists():
        print('\tEncoder model does not exist, need to save it first')
        _, encoder = encode_model_2_tuple(fake_2_tuple, encoder=None)
        encoder.save(str(encoder_path))

    if not decoder_path.exists():
        print('\tDecoder model does not exist, need to save it first')
        encoded_assessment, _ = encode_model_2_tuple(fake_2_tuple, encoder=None)
        _, decoder = decode_model_2_tuple_tpr(encoded_assessment, model_2_tuple_has_weights=has_weights)
        decoder.save(str(decoder_path))

    return tf.keras.models.load_model(str(encoder_path)), tf.keras.models.load_model(str(decoder_path))


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

    return res_assessment


def subsymbolic_aggregation(assessments, linguistic_scale_size):
    tf.compat.v1.disable_eager_execution()
    aggregator_num_experts = 2  # limitation of the model
    aggregator_expected_encoding = MTATask.MTAEncodingType.full_no_weights
    has_weights = False

    models_dir = Path(__file__).parent / 'models'
    aggregator_model_path = models_dir / 'mta_48_bits_frozen.pb'

    encoder, decoder = load_encoder_decoder_models(encoder_path=models_dir / 'encoder.savedmodel',
                                                   decoder_path=models_dir / 'decoder.savedmodel')

    aggregator_infer = get_infer_routine(aggregator_model_path)

    (seq_len, inputs, _), _ = _generate_data(aggregator_expected_encoding,
                                             aggregator_num_experts,
                                             linguistic_scale_size)
    bits_per_number = single_tpr_len(linguistic_scale_size)

    print('Step 2. Process assessments (translate to the distributed form if needed)')

    subsym_start = time.perf_counter()

    assert aggregator_num_experts == 2, 'Pre-trained model can work with only two assessments at once'

    accumulated_assessment = assessments[0]  # this is a seed of our hand-made reduce
    for assessment in assessments[1:]:
        pair = [accumulated_assessment, assessment]

        raw_dataset = [
            [pair, Model2Tuple(term_index=0, alpha=0)]  # we do not care here about expected result
        ]
        _, _ = pack_with_full_mta_encoding(raw_dataset, bits_per_number, inputs, encoder)

        print('Step 3. Aggregate assessments (symbolic-/sub-symbolically)')
        print('\tsub-symbolically')
        outputs = aggregator_infer(inputs=inputs, seq_len=seq_len)
        accumulated_assessment = outputs[0][:, 0]

    print('Step 4. Process assessments (translate from the distributed form if needed)')
    res_assessment, _ = decode_model_2_tuple_tpr(accumulated_assessment,
                                                 decoder=decoder,
                                                 model_2_tuple_has_weights=has_weights)

    print('Step 5. Communicating result')
    print(f'\tResulting assessment: {res_assessment}')

    return res_assessment, time.perf_counter() - subsym_start


class SimpleArgumentParser(Tap):
    is_symbolic: bool = True
    is_subsymbolic: bool = True
    task_description: str = None
    num_assessments: int = 2
    linguistic_scale_size: int = 5


def main(args: SimpleArgumentParser):
    assert args.linguistic_scale_size == 5, 'Fuzzy assessments from scales other than of granularity of 5 are ' \
                                            'not supported'

    print('Step 1. Obtain assessments (read or generate)')
    if args.task_description is None:
        print('\tGenerating assessments...')
        assessments = generate_assessments(args.num_assessments, args.linguistic_scale_size)
    else:
        print('\tLoading assessments...')
        assessments = load_assessments(Path(args.task_description), args.num_assessments, args.linguistic_scale_size)

    print(f'\tAssessments: {", ".join([str(i) for i in assessments])}')

    if args.is_symbolic:
        sym_start = time.perf_counter()
        symbolic_res = symbolic_aggregation(assessments, args.linguistic_scale_size)
        sym_total = time.perf_counter() - sym_start
    if args.is_subsymbolic:
        subsymbolic_res, subsym_total = subsymbolic_aggregation(assessments, args.linguistic_scale_size)

    print('Report:')
    if args.is_symbolic:
        print(f'\t{"Symbolic result:":<30}{str(symbolic_res):>30}')
        print(f'\t{"Symbolic time:":<30}{sym_total:>30}')
    if args.is_subsymbolic:
        print(f'\t{"Subsymbolic result:":<30}{str(subsymbolic_res):>30}')
        print(f'\t{"Subsymbolic time:":<30}{subsym_total:>30}')
    if args.is_symbolic and args.is_subsymbolic:
        decision = 'results are ' + ('equal' if subsymbolic_res == symbolic_res else 'not equal')
        delta = f'{subsym_total / sym_total:.6}x'
        print(f'\t{"Decision:":<30}{decision:>30}')
        print(f'\t{"Symbolic is faster in:":<30}{delta:>30}')


if __name__ == '__main__':
    main(SimpleArgumentParser().parse_args())
