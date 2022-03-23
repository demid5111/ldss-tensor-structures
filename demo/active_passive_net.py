"""
    We have two sentences:
    1. Few leaders are admired by George (Passive voice)
    2. George admires few leaders (Active voice)

    When using the notation from the original paper
    Legendre, G., Miyata, Y., & Smolensky, P. (1991).
    Distributed recursive structure processing.
    In Advances in Neural Information Processing Systems (pp. 591-597).
    Those trees are:

    Active voice
      (root)
    /         \
    A       /     \
          V       P

    Passive voice
      (root)
    /         \
    P       /           \
          /     \      /       \
        Aux     V     by        A


    We need to extract the following structure:

      (root)
    /         \
    V       /  \
          A     P
"""
import numpy as np
import tensorflow as tf

from core.active_passive_net.classifier.vendor.network import build_filler_extractor_network, \
    build_real_filler_extractor_network
from core.active_passive_net.utils import elementary_join
from core.active_passive_net.vendor.network import build_active_passive_network
from core.joiner.vendor.network import build_tree_joiner_network
from core.unshifter.vendor.network import build_tree_unshifter_network
from core.utils import flattenize_per_tensor_representation, get_filler_by
from demo.shifting_structure import generate_shapes, generate_input_placeholder, \
    extract_per_level_tensor_representation_after_shift, reshape_to_satisfy_max_depth_after_shift
from demo.unshifting_structure import reshape_to_satisfy_max_depth_after_unshift, generate_shapes_for_unshift, \
    extract_per_level_tensor_representation_after_unshift


def elementary_extract(extract_network, input_structure_max_shape, basic_roles, basic_fillers, tree):
    # TODO: unify the pipeline with join: do we need to make reshape as the first step of operation or the last

    # max depth of the resulting tree
    max_depth = input_structure_max_shape.shape[0]
    single_role_shape = basic_roles[0].shape
    single_filler_shape = basic_fillers[0].shape
    prepared_for_unshift = reshape_to_satisfy_max_depth_after_unshift(tree,
                                                                      max_depth,
                                                                      single_role_shape,
                                                                      single_filler_shape)

    extracted_child = extract_network.predict_on_batch([
        *prepared_for_unshift
    ])
    extracted_child = extracted_child.reshape((*extracted_child.shape[1:],))

    return extract_per_level_tensor_representation_after_unshift(
        extracted_child,
        max_tree_depth=max_depth,
        role_shape=single_role_shape,
        filler_shape=single_filler_shape)


def check_passive_case(ap_net, encoded_sentence, roles, fillers, dual_roles):
    tree_for_unshift = flattenize_per_tensor_representation(encoded_sentence)
    tree_for_unshift = tree_for_unshift.reshape((1, *tree_for_unshift.shape, 1))
    extracted_semantic_tree = ap_net.predict_on_batch([
        tree_for_unshift
    ])
    tensor_repr = extract_per_level_tensor_representation_after_shift(extracted_semantic_tree,
                                                                      max_tree_depth=2,
                                                                      role_shape=roles[0].shape,
                                                                      filler_shape=fillers[0].shape)
    syntax_tree = reshape_to_satisfy_max_depth_after_shift(tensor_repr,
                                                           3,
                                                           roles[0].shape,
                                                           fillers[0].shape)
    print('Extracted semantic tree')

    keras_decode_verb = build_real_filler_extractor_network(roles=dual_roles,
                                                            fillers=fillers,
                                                            tree_shape=syntax_tree,
                                                            role_extraction_order=[0],
                                                            stop_level=1)

    tree_for_unshift = flattenize_per_tensor_representation(syntax_tree)
    tree_for_unshift = tree_for_unshift.reshape((1, *tree_for_unshift.shape, 1))
    extracted_verb = keras_decode_verb.predict_on_batch([
        tree_for_unshift
    ])

    keras_decode_agent = build_real_filler_extractor_network(roles=dual_roles,
                                                             fillers=fillers,
                                                             tree_shape=syntax_tree,
                                                             role_extraction_order=[1, 0],
                                                             stop_level=1)

    tree_for_unshift = flattenize_per_tensor_representation(syntax_tree)
    tree_for_unshift = tree_for_unshift.reshape((1, *tree_for_unshift.shape, 1))
    extracted_agent = keras_decode_agent.predict_on_batch([
        tree_for_unshift
    ])

    keras_decode_patient = build_real_filler_extractor_network(roles=dual_roles,
                                                               fillers=fillers,
                                                               tree_shape=syntax_tree,
                                                               role_extraction_order=[1, 1],
                                                               stop_level=1)

    tree_for_unshift = flattenize_per_tensor_representation(syntax_tree)
    tree_for_unshift = tree_for_unshift.reshape((1, *tree_for_unshift.shape, 1))
    extracted_patient = keras_decode_patient.predict_on_batch([
        tree_for_unshift
    ])

    return {
        'A': extracted_agent,
        'P': extracted_patient,
        'V': extracted_verb,
    }


def check_active_case(ap_net, encoded_sentence, roles, fillers, dual_roles):
    tree_for_unshift = flattenize_per_tensor_representation(encoded_sentence)
    tree_for_unshift = tree_for_unshift.reshape((1, *tree_for_unshift.shape, 1))
    extracted_semantic_tree = ap_net.predict_on_batch([
        tree_for_unshift
    ])
    tensor_repr = extract_per_level_tensor_representation_after_shift(extracted_semantic_tree,
                                                                      max_tree_depth=2,
                                                                      role_shape=roles[0].shape,
                                                                      filler_shape=fillers[0].shape)
    syntax_tree = reshape_to_satisfy_max_depth_after_shift(tensor_repr,
                                                           3,
                                                           roles[0].shape,
                                                           fillers[0].shape)
    print('Extracted semantic tree')

    keras_decode_verb = build_real_filler_extractor_network(roles=dual_roles,
                                                            fillers=fillers,
                                                            tree_shape=syntax_tree,
                                                            role_extraction_order=[0],
                                                            stop_level=1)

    tree_for_unshift = flattenize_per_tensor_representation(syntax_tree)
    tree_for_unshift = tree_for_unshift.reshape((1, *tree_for_unshift.shape, 1))
    extracted_verb = keras_decode_verb.predict_on_batch([
        tree_for_unshift
    ])

    keras_decode_agent = build_real_filler_extractor_network(roles=dual_roles,
                                                             fillers=fillers,
                                                             tree_shape=syntax_tree,
                                                             role_extraction_order=[1, 0],
                                                             stop_level=1)

    tree_for_unshift = flattenize_per_tensor_representation(syntax_tree)
    tree_for_unshift = tree_for_unshift.reshape((1, *tree_for_unshift.shape, 1))
    extracted_agent = keras_decode_agent.predict_on_batch([
        tree_for_unshift
    ])

    keras_decode_patient = build_real_filler_extractor_network(roles=dual_roles,
                                                               fillers=fillers,
                                                               tree_shape=syntax_tree,
                                                               role_extraction_order=[1, 1],
                                                               stop_level=1)

    tree_for_unshift = flattenize_per_tensor_representation(syntax_tree)
    tree_for_unshift = tree_for_unshift.reshape((1, *tree_for_unshift.shape, 1))
    extracted_patient = keras_decode_patient.predict_on_batch([
        tree_for_unshift
    ])

    return {
        'A': extracted_agent,
        'P': extracted_patient,
        'V': extracted_verb,
    }


def encode_active_voice_sentence(roles, fillers, fillers_order):
    MAX_TREE_DEPTH = 4
    SINGLE_ROLE_SHAPE = roles[0].shape
    SINGLE_FILLER_SHAPE = fillers[0].shape

    fillers_shapes = generate_shapes(max_tree_depth=MAX_TREE_DEPTH,
                                     role_shape=SINGLE_ROLE_SHAPE,
                                     filler_shape=SINGLE_FILLER_SHAPE)

    keras_joiner = build_tree_joiner_network(roles=roles, fillers_shapes=fillers_shapes)

    t_V_r0_P_r1 = elementary_join(joiner_network=keras_joiner,
                                  input_structure_max_shape=fillers_shapes,
                                  basic_roles=roles,
                                  basic_fillers=fillers,
                                  subtrees=(
                                      get_filler_by(name='V', order=fillers_order,
                                                    fillers=fillers),
                                      get_filler_by(name='P', order=fillers_order,
                                                    fillers=fillers)
                                  ))
    print('calculated cons(V,P)')

    t_active_voice = elementary_join(joiner_network=keras_joiner,
                                     input_structure_max_shape=fillers_shapes,
                                     basic_roles=roles,
                                     basic_fillers=fillers,
                                     subtrees=(
                                         get_filler_by(name='A', order=fillers_order, fillers=fillers),
                                         t_V_r0_P_r1
                                     ))
    print('calculated cons(A,cons(V,P))')
    print('Found tensor representation of the Active Voice sentence')
    return t_active_voice


def encode_passive_voice_sentence(roles, fillers, fillers_order):
    MAX_TREE_DEPTH = 4
    SINGLE_ROLE_SHAPE = roles[0].shape
    SINGLE_FILLER_SHAPE = fillers[0].shape

    fillers_shapes = generate_shapes(max_tree_depth=MAX_TREE_DEPTH,
                                     role_shape=SINGLE_ROLE_SHAPE,
                                     filler_shape=SINGLE_FILLER_SHAPE)

    keras_joiner = build_tree_joiner_network(roles=roles, fillers_shapes=fillers_shapes)

    t_by_r0_A_r1 = elementary_join(joiner_network=keras_joiner,
                                   input_structure_max_shape=fillers_shapes,
                                   basic_roles=roles,
                                   basic_fillers=fillers,
                                   subtrees=(
                                       get_filler_by(name='by', order=fillers_order, fillers=fillers),
                                       get_filler_by(name='A', order=fillers_order, fillers=fillers)
                                   ))
    print('calculated cons(by,A)')

    t_Aux_r0_V_r1 = elementary_join(joiner_network=keras_joiner,
                                    input_structure_max_shape=fillers_shapes,
                                    basic_roles=roles,
                                    basic_fillers=fillers,
                                    subtrees=(
                                        get_filler_by(name='Aux', order=fillers_order, fillers=fillers),
                                        get_filler_by(name='V', order=fillers_order, fillers=fillers)
                                    ))
    print('calculated cons(Aux,V)')

    t_Aux_r0r0_V_r1r0_by_r0r1_A_r1r1 = elementary_join(joiner_network=keras_joiner,
                                                       input_structure_max_shape=fillers_shapes,
                                                       basic_roles=roles,
                                                       basic_fillers=fillers,
                                                       subtrees=(
                                                           t_Aux_r0_V_r1,
                                                           t_by_r0_A_r1
                                                       ))
    print('calculated cons(cons(Aux,V), cons(by,A))')

    t_passive_voice = elementary_join(joiner_network=keras_joiner,
                                      input_structure_max_shape=fillers_shapes,
                                      basic_roles=roles,
                                      basic_fillers=fillers,
                                      subtrees=(
                                          get_filler_by(name='P',
                                                        order=fillers_order,
                                                        fillers=fillers),
                                          t_Aux_r0r0_V_r1r0_by_r0r1_A_r1r1
                                      ))
    print('calculated cons(P, cons(cons(Aux,V), cons(by,A)))')
    print('Found tensor representation of the Passive Voice sentence')
    return t_passive_voice


if __name__ == '__main__':
    print('Hello, Active-Passive Net')
    tf.compat.v1.disable_eager_execution()

    # Input information
    fillers = np.array([
        [7, 0, 0, 0, 0],  # A
        [0, 4, 0, 0, 0],  # V
        [0, 0, 2, 0, 0],  # P
        [0, 0, 0, 5, 0],  # Aux
        [0, 0, 0, 0, 3],  # by
    ])
    roles = np.array([
        [10, 0],  # r_0
        [0, 5],  # r_1
    ])
    dual_basic_roles_case_1 = np.linalg.inv(roles)
    mapping_case_active = {
        'A': [0],
        'V': [0, 1],
        'P': [1, 1]
    }
    order_case_active = ['A', 'V', 'P']

    mapping_case_passive = {
        'P': [0],
        'Aux': [0, 0, 1],
        'V': [1, 0, 1],
        'by': [0, 1, 1],
        'A': [1, 1, 1],
    }
    order_case_passive = ['A', 'V', 'P', 'Aux', 'by']

    MAX_TREE_DEPTH = 4
    SINGLE_ROLE_SHAPE = roles[0].shape
    SINGLE_FILLER_SHAPE = fillers[0].shape

    t_active_voice = encode_active_voice_sentence(roles=roles,
                                                  fillers=fillers,
                                                  fillers_order=order_case_active)
    t_passive_voice = encode_passive_voice_sentence(roles=roles,
                                                    fillers=fillers,
                                                    fillers_order=order_case_passive)

    fillers_shapes_unshift = generate_shapes_for_unshift(max_tree_depth=MAX_TREE_DEPTH - 1,
                                                         role_shape=SINGLE_ROLE_SHAPE,
                                                         filler_shape=SINGLE_FILLER_SHAPE)
    keras_ex1_unshifter = build_tree_unshifter_network(roles=dual_basic_roles_case_1,
                                                       fillers_shapes=fillers_shapes_unshift,
                                                       role_index=1)
    keras_ex0_unshifter = build_tree_unshifter_network(roles=dual_basic_roles_case_1,
                                                       fillers_shapes=fillers_shapes_unshift,
                                                       role_index=0)

    t_Aux_r0r0_V_r1r0_by_r0r1_A_r1r1 = elementary_extract(extract_network=keras_ex1_unshifter,
                                                          input_structure_max_shape=fillers_shapes_unshift,
                                                          basic_roles=roles,
                                                          basic_fillers=fillers,
                                                          tree=t_passive_voice)

    t_Aux_r0_V_r1 = elementary_extract(extract_network=keras_ex0_unshifter,
                                       input_structure_max_shape=fillers_shapes_unshift,
                                       basic_roles=roles,
                                       basic_fillers=fillers,
                                       tree=t_Aux_r0r0_V_r1r0_by_r0r1_A_r1r1)

    t_Aux = elementary_extract(extract_network=keras_ex0_unshifter,
                               input_structure_max_shape=fillers_shapes_unshift,
                               basic_roles=roles,
                               basic_fillers=fillers,
                               tree=t_Aux_r0_V_r1)

    print(t_Aux)

    keras_full_unshifter = build_filler_extractor_network(roles=dual_basic_roles_case_1,
                                                          fillers=fillers,
                                                          tree_shape=t_passive_voice,
                                                          role_extraction_order=[1, 0, 0],
                                                          stop_level=0)

    tree_for_unshift = flattenize_per_tensor_representation(t_passive_voice)
    tree_for_unshift = tree_for_unshift.reshape((1, *tree_for_unshift.shape, 1))
    extracted_child = keras_full_unshifter.predict_on_batch([
        tree_for_unshift
    ])

    print(extracted_child)

    tree_for_unshift = flattenize_per_tensor_representation(t_active_voice)
    tree_for_unshift = tree_for_unshift.reshape((1, *tree_for_unshift.shape, 1))
    extracted_child = keras_full_unshifter.predict_on_batch([
        tree_for_unshift
    ])

    print(extracted_child)

    ##########################
    # REAL ACTIVE-PASSIVE NETWORK

    # Extraction head
    print('Started structure manipulations')

    keras_active_passive_network = build_active_passive_network(roles=roles,
                                                                dual_roles=dual_basic_roles_case_1,
                                                                fillers=fillers,
                                                                tree_shape=t_passive_voice)

    check_passive_case(ap_net=keras_active_passive_network,
                       encoded_sentence=t_passive_voice,
                       roles=roles,
                       fillers=fillers,
                       dual_roles=dual_basic_roles_case_1)

    check_active_case(ap_net=keras_active_passive_network,
                      encoded_sentence=t_active_voice,
                      roles=roles,
                      fillers=fillers,
                      dual_roles=dual_basic_roles_case_1)
