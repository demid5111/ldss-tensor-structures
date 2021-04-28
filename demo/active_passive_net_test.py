import unittest
import numpy as np
import tensorflow as tf

from core.active_passive_net.vendor.network import build_active_passive_network
from core.joiner.vendor.network import build_tree_joiner_network
from demo.active_passive_net import elementary_join, get_filler_by, check_active_case, encode_active_voice_sentence, \
    encode_passive_voice_sentence, check_passive_case
from demo.shifting_structure import generate_shapes

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
order_case_active = ['A', 'V', 'P']
order_case_passive = ['A', 'V', 'P', 'Aux', 'by']
dual_basic_roles_case_1 = np.linalg.inv(roles)
SINGLE_ROLE_SHAPE = roles[0].shape
SINGLE_FILLER_SHAPE = fillers[0].shape


class ElementaryJoinTest(unittest.TestCase):
    def test_ideal1(self):
        """
        First use case for the structure that has nesting equal 1

        root
        | \  \
        A B  C
        """
        # Input information

        MAX_TREE_DEPTH = 3

        fillers_shapes = generate_shapes(max_tree_depth=MAX_TREE_DEPTH,
                                         role_shape=SINGLE_ROLE_SHAPE,
                                         filler_shape=SINGLE_FILLER_SHAPE)

        keras_joiner = build_tree_joiner_network(roles=roles, fillers_shapes=fillers_shapes)

        t_V_r0_P_r1 = elementary_join(joiner_network=keras_joiner,
                                      input_structure_max_shape=fillers_shapes,
                                      basic_roles=roles,
                                      basic_fillers=fillers,
                                      subtrees=(
                                          get_filler_by(name='V', order=order_case_active, fillers=fillers),
                                          get_filler_by(name='P', order=order_case_active, fillers=fillers)
                                      ))

        t_V_r0_P_r1_expected = [
            np.array([
                [
                    0.,
                    0.,
                    0.,
                    0.,
                    0.,
                ]
            ], dtype=np.float32),
            np.array([
                [
                    [0., 0.],
                    [40., 0.],
                    [0., 10.],
                    [0., 0.],
                    [0., 0.],
                ]
            ], dtype=np.float32),
            np.array([
                [
                    [
                        [0., 0.],
                        [0., 0.]
                    ],
                    [
                        [0., 0.],
                        [0., 0.]
                    ],
                    [
                        [0., 0.],
                        [0., 0.]
                    ],
                    [
                        [0., 0.],
                        [0., 0.]
                    ],
                    [
                        [0., 0.],
                        [0., 0.]
                    ]
                ]
            ], dtype=np.float32)
        ]
        for level_idx in range(len(t_V_r0_P_r1)):
            np.testing.assert_array_equal(
                t_V_r0_P_r1[level_idx],
                t_V_r0_P_r1_expected[level_idx]
            )


class APNETIntegrationTest(unittest.TestCase):
    def test_active_voice_sentence_ideal(self):
        tf.compat.v1.disable_eager_execution()
        t_active_voice = encode_active_voice_sentence(roles=roles,
                                                      fillers=fillers,
                                                      fillers_order=order_case_active)

        keras_active_passive_network = build_active_passive_network(roles=roles,
                                                                    dual_roles=dual_basic_roles_case_1,
                                                                    fillers=fillers,
                                                                    tree_shape=t_active_voice)

        res = check_active_case(ap_net=keras_active_passive_network,
                                encoded_sentence=t_active_voice,
                                roles=roles,
                                fillers=fillers,
                                dual_roles=dual_basic_roles_case_1)

        for filler_name, filler_raw_encoded in res.items():
            expected_filler_value = get_filler_by(name=filler_name, order=order_case_active, fillers=fillers)
            filler_encoded = np.reshape(filler_raw_encoded[:len(expected_filler_value)], expected_filler_value.shape)
            np.testing.assert_array_almost_equal(expected_filler_value, filler_encoded)

    def test_passive_voice_sentence_ideal(self):
        tf.compat.v1.disable_eager_execution()
        t_passive_voice = encode_passive_voice_sentence(roles=roles,
                                                        fillers=fillers,
                                                        fillers_order=order_case_passive)

        keras_active_passive_network = build_active_passive_network(roles=roles,
                                                                    dual_roles=dual_basic_roles_case_1,
                                                                    fillers=fillers,
                                                                    tree_shape=t_passive_voice)

        res = check_passive_case(ap_net=keras_active_passive_network,
                                 encoded_sentence=t_passive_voice,
                                 roles=roles,
                                 fillers=fillers,
                                 dual_roles=dual_basic_roles_case_1)

        for filler_name, filler_raw_encoded in res.items():
            expected_filler_value = get_filler_by(name=filler_name, order=order_case_active, fillers=fillers)
            filler_encoded = np.reshape(filler_raw_encoded[:len(expected_filler_value)], expected_filler_value.shape)
            np.testing.assert_array_almost_equal(expected_filler_value, filler_encoded)
