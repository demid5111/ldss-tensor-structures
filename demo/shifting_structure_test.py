import unittest
import numpy as np

from core.joiner.vendor.network import build_tree_joiner_network
from demo.shifting_structure import generate_shapes, generate_input_placeholder, extract_per_level_tensor_representation_after_shift


class ShiftingIntegrationTest(unittest.TestCase):
    def test_ideal1(self):
        """
        First use case for the structure that should be shifted left

        Starting from:
        root

        We want to get:
        root
        |
        A (left-child-of-root)
        """
        fillers_case_1 = np.array([
            [8, 0, 0],  # A
            [0, 15, 0],  # B
            [0, 0, 10],  # C
        ])
        roles_case_1 = np.array([
            [10, 0],  # r_0
            [0, 5],  # r_1
        ])

        MAX_TREE_DEPTH = 2
        SINGLE_ROLE_SHAPE = roles_case_1[0].shape
        SINGLE_FILLER_SHAPE = fillers_case_1[0].shape

        fillers_shapes = generate_shapes(max_tree_depth=MAX_TREE_DEPTH,
                                         role_shape=SINGLE_ROLE_SHAPE,
                                         filler_shape=SINGLE_FILLER_SHAPE)

        keras_joiner = build_tree_joiner_network(roles=roles_case_1, fillers_shapes=fillers_shapes)

        left_subtree_placeholder = generate_input_placeholder(fillers_shapes)
        left_subtree_placeholder[0] = fillers_case_1[0].reshape(1, *SINGLE_FILLER_SHAPE)
        right_subtree_placeholder = generate_input_placeholder(fillers_shapes)

        fillers_joined = keras_joiner.predict_on_batch([
            *left_subtree_placeholder,
            *right_subtree_placeholder
        ])

        expected_subtree = np.array(
            [
                [80.],
                [0.],
                [0.],
                [0.],
                [0.],
                [0.],
                [0.],
                [0.],
                [0.],
                [0.],
                [0.],
                [0.],
                [0.],
                [0.],
                [0.],
                [0.],
                [0.],
                [0.]
            ]
        )
        np.testing.assert_array_equal(fillers_joined, expected_subtree)


class ExtractTensorsTest(unittest.TestCase):
    def test_ideal1(self):
        input_subtree = np.array(
            [
                [80.],
                [0.],
                [0.],
                [0.],
                [0.],
                [0.],
                [0.],
                [0.],
                [0.],
                [0.],
                [0.],
                [0.],
                [0.],
                [0.],
                [0.],
                [0.],
                [0.],
                [0.]
            ]
        )

        fillers_case_1 = np.array([
            [8, 0, 0],  # A
            [0, 15, 0],  # B
            [0, 0, 10],  # C
        ])
        roles_case_1 = np.array([
            [10, 0],  # r_0
            [0, 5],  # r_1
        ])

        MAX_TREE_DEPTH = 2
        SINGLE_ROLE_SHAPE = roles_case_1[0].shape
        SINGLE_FILLER_SHAPE = fillers_case_1[0].shape

        tensor_repr_A_x_r_0 = extract_per_level_tensor_representation_after_shift(input_subtree,
                                                                                  max_tree_depth=MAX_TREE_DEPTH,
                                                                                  role_shape=SINGLE_ROLE_SHAPE,
                                                                                  filler_shape=SINGLE_FILLER_SHAPE)

        tensor_repr_A_x_r_0_expected = [
            np.array([
                [
                    [80., 0.],
                    [0., 0.],
                    [0., 0.]
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
                    ]
                ]
            ], dtype=np.float32)
        ]
        for level_idx in range(len(tensor_repr_A_x_r_0)):
            np.testing.assert_array_equal(
                tensor_repr_A_x_r_0[level_idx],
                tensor_repr_A_x_r_0_expected[level_idx]
            )
