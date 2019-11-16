import unittest
import numpy as np

from core.joiner.vendor.network import build_tree_joiner_network
from demo.active_passive_net import elementary_join, get_filler_by
from demo.shifting_structure import generate_shapes


class ElementaryJoinTest(unittest.TestCase):
    def test_ideal1(self):
        """
        First use case for the structure that has nesting equal 1

        root
        | \  \
        A B  C
        """
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
        order_case_active = ['A', 'V', 'P']

        MAX_TREE_DEPTH = 3
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
