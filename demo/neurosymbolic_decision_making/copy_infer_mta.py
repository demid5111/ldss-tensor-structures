from pathlib import Path

import tensorflow as tf

from demo.neurosymbolic_decision_making.copy_generator import MTATaskData
from demo.neurosymbolic_decision_making.copy_task import MTATask


def _generate_data(mta_encoding, num_experts, scale_size):
    generator_args = dict(
        num_batches=1,
        batch_size=32,
        bits_per_vector=3,
        curriculum_point=None,
        max_seq_len=-1,  # made intentionally, generator will define TPR length itself
        curriculum='none',
        pad_to_max_seq_len=False
    )

    generator_args['cli_mode'] = mta_encoding in (
        MTATask.MTAEncodingType.full, MTATask.MTAEncodingType.full_no_weights)
    generator_args['numbers_quantity'] = num_experts
    generator_args['two_tuple_weight_precision'] = 1
    generator_args['two_tuple_alpha_precision'] = 1
    generator_args['two_tuple_largest_scale_size'] = scale_size
    generator_args['mta_encoding'] = mta_encoding

    data_generator = MTATaskData()

    return data_generator.generate_batches(**generator_args)[0], data_generator


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.compat.v2.io.gfile.GFile(frozen_graph_filename, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name='prefix')
    return graph


def prepare_graph_for_inference(model_path: Path, prefix: str = ''):
    graph = load_graph(str(model_path))

    additional_prefix = f'{prefix}/' if prefix else ''
    max_seq_len_placeholder_name = f'{additional_prefix}prefix/root/Placeholder:0'
    inputs_placeholder_name = f'{additional_prefix}prefix/root/Placeholder_1:0'
    output_name = f'{additional_prefix}prefix/root/Sigmoid:0'

    inputs_placeholder = graph.get_tensor_by_name(inputs_placeholder_name)
    max_seq_len_placeholder = graph.get_tensor_by_name(max_seq_len_placeholder_name)

    y = graph.get_tensor_by_name(output_name)

    return graph, (
        inputs_placeholder,
        max_seq_len_placeholder
    ), y


def infer_model(model_path: Path, inputs, seq_len):
    graph, (inputs_placeholder, seq_len_placeholder), y = prepare_graph_for_inference(model_path, prefix='prefix')
    with tf.compat.v1.Session(graph=graph) as sess:
        outputs = sess.run(y, feed_dict={
            inputs_placeholder: inputs,
            seq_len_placeholder: seq_len
        })
    return outputs
