import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model

from core.active_passive_net.classifier.vendor.network import build_universal_extraction_branch
from core.unshifter.vendor.network import unshift_matrix


def build_decode_model_2_tuple_network(filler_len, dual_roles, max_depth):
    input_num_elements, flattened_tree_num_elements = unshift_matrix(dual_roles[0], filler_len, max_depth - 1).shape
    shape = (flattened_tree_num_elements + filler_len, 1)
    flattened_input = Input(shape=(*shape,), batch_shape=(*shape,))

    index_const_inputs, index_raw_output, _ = build_universal_extraction_branch(model_input=flattened_input,
                                                                                roles=dual_roles,
                                                                                filler_len=filler_len,
                                                                                max_depth=max_depth - 1,
                                                                                stop_level=max_depth - 1,
                                                                                role_extraction_order=[0],
                                                                                prefix='extracting_index')
    index_raw_output = Lambda(lambda x: K.tf.reshape(x, (filler_len,)))(index_raw_output)

    alpha_const_inputs, alpha_raw_output, _ = build_universal_extraction_branch(model_input=flattened_input,
                                                                                roles=dual_roles,
                                                                                filler_len=filler_len,
                                                                                max_depth=max_depth - 1,
                                                                                stop_level=max_depth - 1,
                                                                                role_extraction_order=[1],
                                                                                prefix='extracting_alpha')
    alpha_raw_output = Lambda(lambda x: K.tf.reshape(x, (filler_len,)))(alpha_raw_output)

    weight_const_inputs, weight_raw_output, _ = build_universal_extraction_branch(model_input=flattened_input,
                                                                                  roles=dual_roles,
                                                                                  filler_len=filler_len,
                                                                                  max_depth=max_depth - 1,
                                                                                  stop_level=max_depth - 1,
                                                                                  role_extraction_order=[2],
                                                                                  prefix='extracting_weight')
    weight_raw_output = Lambda(lambda x: K.tf.reshape(x, (filler_len,)))(weight_raw_output)

    return Model(
        inputs=[
            *index_const_inputs,
            *alpha_const_inputs,
            *weight_const_inputs,
            flattened_input
        ],
        outputs=[
            index_raw_output,
            alpha_raw_output,
            weight_raw_output
        ]
    )
