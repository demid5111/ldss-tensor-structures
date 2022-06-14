import tensorflow as tf

from core.peano.increment.vendor.network import check_if_not_zero_branch
from core.unshifter.vendor.network import unshift_matrix


def build_decode_number_network(fillers, dual_roles, max_depth):
    filler_len = fillers[0].shape[0]

    input_num_elements, flattened_tree_num_elements = unshift_matrix(dual_roles[0], filler_len, max_depth - 1).shape
    shape = (flattened_tree_num_elements + filler_len, 1)
    flattened_input = tf.keras.layers.Input(shape=(*shape,), batch_size=1)

    is_not_zero, unshifted = check_if_not_zero_branch(
        decrementing_input=flattened_input,
        role=dual_roles[0],
        filler_len=filler_len,
        max_depth=max_depth,
        block_id=1)

    return tf.keras.Model(
        inputs=[
            flattened_input
        ],
        outputs=[
            unshifted,
            is_not_zero
        ]
    )
