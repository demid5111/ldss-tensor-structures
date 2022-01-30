import tensorflow as tf


def mat_mul(tensors):
    return [tf.keras.backend.dot(tensors[0][i],tensors[1][i]) for i in range(tensors[0].shape[0])]


def mat_transpose(matrix):
    return tf.keras.backend.permute_dimensions(matrix, (1,0))


def prepare_shapes(mapping, dual_roles):
    number_roles = len(list(mapping.values()))
    return 1, number_roles, dual_roles.shape[1]


def build_filler_decoder_network(input_shapes):
    transposer = tf.keras.layers.Permute((2, 1))
    transposer_matrix = tf.keras.layers.Lambda(mat_transpose)
    unbinding_cell = tf.keras.layers.Lambda(mat_mul)

    tensor_representation_shape = input_shapes[0]
    dual_roles_shape = input_shapes[1]

    input_tensor_representation_layer = tf.keras.layers.Input(shape=tensor_representation_shape[1:], batch_size=1)
    input_dual_roles_layer = tf.keras.layers.Input(shape=dual_roles_shape[1:], batch_size=1)

    transposed_dual_role_layer = transposer(input_dual_roles_layer)
    binding_tensors_layer = unbinding_cell([
        input_tensor_representation_layer,
        transposed_dual_role_layer
    ])

    transposed_fillers = transposer_matrix(binding_tensors_layer[0])

    return tf.keras.Model(
        inputs=[
            input_tensor_representation_layer,
            input_dual_roles_layer
        ],
        outputs=transposed_fillers)
