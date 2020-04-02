import unittest
from typing import Tuple

import numpy as np

from core.joiner.vendor.network import build_tree_joiner_network
from core.peano.increment.vendor.network import build_increment_network
from demo.active_passive_net import flattenize_per_tensor_representation
from demo.peano_mtwa_net import number_to_tree, get_max_tree_depth, decode_number
from demo.shifting_structure import generate_shapes
from demo.unshifting_structure import extract_per_level_tensor_representation_after_unshift

fillers = np.array([
    [7, 0, 0, 0, 0],  # A
])

roles = np.array([
    [10, 0],  # r_0
    [0, 5],  # r_1
])
dual_basic_roles_case_1 = np.linalg.inv(roles)
order_case_active = ['A', ]
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
    def sum(self, a, b, max_number=None) -> Tuple[any, int]:
        if max_number is None:
            max_number = a + b

        max_tree_depth = get_max_tree_depth(max_number)  # as one is already represented by structure with 2 levels

        # Input information
        fillers_shapes = generate_shapes(max_tree_depth=max_tree_depth,
                                         role_shape=SINGLE_ROLE_SHAPE,
                                         filler_shape=SINGLE_FILLER_SHAPE)

        keras_joiner = build_tree_joiner_network(roles=roles, fillers_shapes=fillers_shapes)

        one_tree = number_to_tree(1, keras_joiner, fillers_shapes, fillers, roles, order_case_active)
        one_tree_flattened = flattenize_per_tensor_representation(one_tree)

        a_tree = number_to_tree(a, keras_joiner, fillers_shapes, fillers, roles, order_case_active)
        a_tree_flattened = flattenize_per_tensor_representation(a_tree)

        b_tree = number_to_tree(b, keras_joiner, fillers_shapes, fillers, roles, order_case_active)
        b_tree_flattened = flattenize_per_tensor_representation(b_tree)

        keras_increment_network = build_increment_network(roles=roles,
                                                          fillers=fillers,
                                                          dual_roles=dual_basic_roles_case_1,
                                                          max_depth=max_tree_depth,
                                                          increment_value=one_tree_flattened)

        new_number = keras_increment_network.predict_on_batch([
            a_tree_flattened,
            b_tree_flattened
        ])

        return extract_per_level_tensor_representation_after_unshift(new_number, max_tree_depth,
                                                                     SINGLE_ROLE_SHAPE,
                                                                     SINGLE_FILLER_SHAPE), \
               max_tree_depth

    def test_1_plus_1(self):
        new_number_tree, max_depth = self.sum(1, 1)

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

    def test_0_plus_1(self):
        new_number_tree, max_depth = self.sum(0, 1)

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

    def test_1_plus_0(self):
        new_number_tree, max_depth = self.sum(1, 0)

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

    def test_2_plus_2(self):
        new_number_tree, max_depth = self.sum(2, 2)

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
            ], dtype=np.float32),
            np.array([
                [
                    [
                        [
                            [
                                [0., 0.],
                                [0., 0.]
                            ],
                            [
                                [0., 0.],
                                [0., 0.]
                            ]
                        ],
                        [
                            [
                                [0., 0.],
                                [0., 0.]
                            ],
                            [
                                [0., 0.],
                                [0., 0.]
                            ]
                        ],

                    ],
                    [
                        [
                            [
                                [0., 0.],
                                [0., 0.]
                            ],
                            [
                                [0., 0.],
                                [0., 0.]
                            ]
                        ],
                        [
                            [
                                [0., 0.],
                                [0., 0.]
                            ],
                            [
                                [0., 0.],
                                [0., 0.]
                            ]
                        ],

                    ],
                    [
                        [
                            [
                                [0., 0.],
                                [0., 0.]
                            ],
                            [
                                [0., 0.],
                                [0., 0.]
                            ]
                        ],
                        [
                            [
                                [0., 0.],
                                [0., 0.]
                            ],
                            [
                                [0., 0.],
                                [0., 0.]
                            ]
                        ],

                    ],
                    [
                        [
                            [
                                [0., 0.],
                                [0., 0.]
                            ],
                            [
                                [0., 0.],
                                [0., 0.]
                            ]
                        ],
                        [
                            [
                                [0., 0.],
                                [0., 0.]
                            ],
                            [
                                [0., 0.],
                                [0., 0.]
                            ]
                        ],

                    ],
                    [
                        [
                            [
                                [0., 0.],
                                [0., 0.]
                            ],
                            [
                                [0., 0.],
                                [0., 0.]
                            ]
                        ],
                        [
                            [
                                [0., 0.],
                                [0., 0.]
                            ],
                            [
                                [0., 0.],
                                [0., 0.]
                            ]
                        ],

                    ],
                ]
            ], dtype=np.float32)
        ]

        self.assertTensorsEqual(expected, new_number_tree)

        result_number = decode_number(number_tree=new_number_tree,
                                      fillers=fillers,
                                      dual_roles=dual_basic_roles_case_1,
                                      max_depth=max_depth)
        self.assertEqual(expected_numeric, result_number)
