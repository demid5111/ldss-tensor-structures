import tensorflow as tf


def mul_vec_on_vec(tensors):
    return [tensors[0][i] * tensors[1][i] for i in range(tensors[0][0].shape[0])]


def prepare_shapes(filler_vectors, roles_vectors):
    return (*filler_vectors.shape, 1), (*roles_vectors.shape, 1)


def build_encoder_network(input_shapes):
    transposer = tf.keras.layers.Permute((2, 1))
    binding_cell = tf.keras.layers.Lambda(mul_vec_on_vec)

    fillers_shape = input_shapes[0]
    roles_shape = input_shapes[1]

    input_fillers_layer = tf.keras.layers.Input(shape=fillers_shape[1:])
    input_roles_layer = tf.keras.layers.Input(shape=roles_shape[1:])

    transposed_role_layer = transposer(input_roles_layer)
    binding_tensors_layer = binding_cell([input_fillers_layer, transposed_role_layer])

    summed_bindings = tf.keras.layers.Add()(binding_tensors_layer)
    # activation = Activation('relu')(summed_bindings)

    return tf.keras.Model(inputs=[input_fillers_layer, input_roles_layer], outputs=summed_bindings)
