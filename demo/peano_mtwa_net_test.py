import unittest
from typing import Tuple

import numpy as np

from core.peano.increment.vendor.network import build_increment_network
from core.peano.utils import number_to_tree, get_max_tree_depth
from demo.active_passive_net import flattenize_per_tensor_representation
from demo.peano_mtwa_net import decode_number, sum_numbers
from demo.unshifting_structure import extract_per_level_tensor_representation_after_unshift

fillers = np.array([
    [7, 0, 0, 0, 0],  # A
])

roles = np.array([
    [10, 0],  # r_0
    [0, 5],  # r_1
])
dual_basic_roles_case_1 = np.linalg.inv(roles)
SINGLE_ROLE_SHAPE = roles[0].shape
SINGLE_FILLER_SHAPE = fillers[0].shape


class TensorAssertions:
    def assertTensorsEqual(self, expected, actual):
        assert len(expected) == len(actual)
        for level_idx in range(len(actual)):
            np.testing.assert_array_equal(
                actual[level_idx],
                expected[level_idx]
            )


class IncrementTest(unittest.TestCase, TensorAssertions):
    @staticmethod
    def increment(a, max_number=None) -> Tuple[any, int]:
        if max_number is None:
            max_number = a + 1

        max_tree_depth = get_max_tree_depth(max_number)  # as one is already represented by structure with 2 levels

        a_tree = number_to_tree(a, max_tree_depth, fillers, roles)
        a_tree_flattened = flattenize_per_tensor_representation(a_tree)

        keras_increment_network = build_increment_network(roles=roles,
                                                          dual_roles=dual_basic_roles_case_1,
                                                          fillers=fillers,
                                                          max_depth=max_tree_depth)

        new_number = keras_increment_network.predict_on_batch([
            a_tree_flattened
        ])

        return (
            extract_per_level_tensor_representation_after_unshift(new_number, max_tree_depth,
                                                                  SINGLE_ROLE_SHAPE,
                                                                  SINGLE_FILLER_SHAPE),
            max_tree_depth
        )

    def test_increment_0(self):
        new_number_tree, max_depth = self.increment(0, max_number=1)

        expected_numeric = 1
        expected = [
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
                    [0., 35.],
                    [0., 0.],
                    [0., 0.],
                    [0., 0.],
                    [0., 0.],
                ]
            ], dtype=np.float32)
        ]

        self.assertTensorsEqual(expected, new_number_tree)

        result_number = decode_number(number_tree=new_number_tree,
                                      fillers=fillers,
                                      dual_roles=dual_basic_roles_case_1,
                                      max_depth=max_depth)
        self.assertEqual(expected_numeric, result_number)

    def test_increment_1(self):
        new_number_tree, max_depth = self.increment(1)

        expected_numeric = 2
        expected = [
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
                    [0., 0.],
                    [0., 0.],
                    [0., 0.],
                    [0., 0.],
                ]
            ], dtype=np.float32),
            np.array([
                [
                    [
                        [0., 0.],
                        [350., 175.]
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

        self.assertTensorsEqual(expected, new_number_tree)

        result_number = decode_number(number_tree=new_number_tree,
                                      fillers=fillers,
                                      dual_roles=dual_basic_roles_case_1,
                                      max_depth=max_depth)
        self.assertEqual(expected_numeric, result_number)

    def test_increment_2(self):
        new_number_tree, max_depth = self.increment(2)

        expected_numeric = 3
        expected = [
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
                    [0., 0.],
                    [0., 0.],
                    [0., 0.],
                    [0., 0.],
                ]
            ], dtype=np.float32),
            np.array([
                [
                    [
                        [0., 0.],
                        [0., 175.]
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
            ], dtype=np.float32),
            np.array([
                [
                    [

                        [
                            [0., 0.],
                            [0., 0.]
                        ],
                        [
                            [3500., 0.],
                            [1750., 0.]
                        ],

                    ],
                    [

                        [
                            [0., 0.],
                            [0., 0.]
                        ],
                        [
                            [0., 0.],
                            [0., 0.]
                        ],

                    ],
                    [

                        [
                            [0., 0.],
                            [0., 0.]
                        ],
                        [
                            [0., 0.],
                            [0., 0.]
                        ],

                    ],
                    [

                        [
                            [0., 0.],
                            [0., 0.]
                        ],
                        [
                            [0., 0.],
                            [0., 0.]
                        ],

                    ],
                    [

                        [
                            [0., 0.],
                            [0., 0.]
                        ],
                        [
                            [0., 0.],
                            [0., 0.]
                        ],

                    ]
                ]
            ], dtype=np.float32)
        ]

        self.assertTensorsEqual(expected, new_number_tree)

        result_number = decode_number(number_tree=new_number_tree,
                                      fillers=fillers,
                                      dual_roles=dual_basic_roles_case_1,
                                      max_depth=max_depth)
        self.assertEqual(expected_numeric, result_number)


class IncrementSingleSum(unittest.TestCase, TensorAssertions):
    def test_single_sum_2_1(self):
        max_number = 3
        max_tree_depth = get_max_tree_depth(max_number)
        a = 2
        b = 1
        c, d = sum_numbers(a, b, max_tree_depth, roles, dual_basic_roles_case_1, fillers, number_sum_blocks=1)
        self.assertEqual(1, c)
        self.assertEqual(2, d)

    def test_single_sum_1_2(self):
        max_number = 3
        max_tree_depth = get_max_tree_depth(max_number)
        a = 1
        b = 2
        c, d = sum_numbers(a, b, max_tree_depth, roles, dual_basic_roles_case_1, fillers, number_sum_blocks=1)
        self.assertEqual(0, c)
        self.assertEqual(3, d)

    def test_single_sum_0_2(self):
        max_number = 3
        max_tree_depth = get_max_tree_depth(max_number)
        a = 0
        b = 2
        c, d = sum_numbers(a, b, max_tree_depth, roles, dual_basic_roles_case_1, fillers, number_sum_blocks=1)
        self.assertEqual(0, c)
        self.assertEqual(2, d)

    def test_single_sum_2_0(self):
        max_number = 3
        max_tree_depth = get_max_tree_depth(max_number)
        a = 2
        b = 0
        c, d = sum_numbers(a, b, max_tree_depth, roles, dual_basic_roles_case_1, fillers, number_sum_blocks=1)
        self.assertEqual(1, c)
        self.assertEqual(1, d)


class IncrementFullSum(unittest.TestCase, TensorAssertions):
    def test_sum_2_1(self):
        max_number = 3
        max_tree_depth = get_max_tree_depth(max_number)
        a = 2
        b = 1
        c, d = sum_numbers(a, b, max_tree_depth, roles, dual_basic_roles_case_1, fillers, number_sum_blocks=max_number)
        self.assertEqual(0, c)
        self.assertEqual(3, d)

    def test_sum_1_2(self):
        max_number = 3
        max_tree_depth = get_max_tree_depth(max_number)
        a = 1
        b = 2
        c, d = sum_numbers(a, b, max_tree_depth, roles, dual_basic_roles_case_1, fillers, number_sum_blocks=max_number)
        self.assertEqual(0, c)
        self.assertEqual(3, d)

    def test_sum_0_2(self):
        max_number = 3
        max_tree_depth = get_max_tree_depth(max_number)
        a = 0
        b = 2
        c, d = sum_numbers(a, b, max_tree_depth, roles, dual_basic_roles_case_1, fillers, number_sum_blocks=max_number)
        self.assertEqual(0, c)
        self.assertEqual(2, d)

    def test_sum_2_0(self):
        max_number = 3
        max_tree_depth = get_max_tree_depth(max_number)
        a = 2
        b = 0
        c, d = sum_numbers(a, b, max_tree_depth, roles, dual_basic_roles_case_1, fillers, number_sum_blocks=max_number)
        self.assertEqual(0, c)
        self.assertEqual(2, d)

    def test_sum_3_0(self):
        max_number = 3
        max_tree_depth = get_max_tree_depth(max_number)
        a = 3
        b = 0
        c, d = sum_numbers(a, b, max_tree_depth, roles, dual_basic_roles_case_1, fillers, number_sum_blocks=max_number)
        self.assertEqual(0, c)
        self.assertEqual(3, d)

    def test_sum_0_3(self):
        max_number = 3
        max_tree_depth = get_max_tree_depth(max_number)
        a = 0
        b = 3
        c, d = sum_numbers(a, b, max_tree_depth, roles, dual_basic_roles_case_1, fillers, number_sum_blocks=max_number)
        self.assertEqual(0, c)
        self.assertEqual(3, d)
