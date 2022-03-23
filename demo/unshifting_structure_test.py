import unittest
import numpy as np
import tensorflow as tf

from core.joiner.vendor.network import build_tree_joiner_network
from core.unshifter.vendor.network import build_tree_unshifter_network
from demo.active_passive_net import elementary_join, get_filler_by
from demo.shifting_structure import generate_shapes
from demo.unshifting_structure import generate_shapes_for_unshift, reshape_to_satisfy_max_depth_after_unshift, \
    extract_per_level_tensor_representation_after_unshift


class UnshiftingIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        tf.compat.v1.disable_eager_execution()

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
        t_active_voice = elementary_join(joiner_network=keras_joiner,
                                         input_structure_max_shape=fillers_shapes,
                                         basic_roles=roles,
                                         basic_fillers=fillers,
                                         subtrees=(
                                             get_filler_by(name='A', order=order_case_active, fillers=fillers),
                                             t_V_r0_P_r1
                                         ))

        dual_basic_roles_case_1 = np.linalg.inv(roles)
        fillers_shapes_unshift = generate_shapes_for_unshift(max_tree_depth=MAX_TREE_DEPTH - 1,
                                                             role_shape=SINGLE_ROLE_SHAPE,
                                                             filler_shape=SINGLE_FILLER_SHAPE)
        keras_ex1_unshifter = build_tree_unshifter_network(roles=dual_basic_roles_case_1,
                                                           fillers_shapes=fillers_shapes_unshift,
                                                           role_index=1)

        prepared_for_unshift = reshape_to_satisfy_max_depth_after_unshift(t_active_voice,
                                                                          MAX_TREE_DEPTH - 1,
                                                                          SINGLE_ROLE_SHAPE,
                                                                          SINGLE_FILLER_SHAPE)

        extracted_child = keras_ex1_unshifter.predict_on_batch([
            *prepared_for_unshift
        ])

        extracted_child = extracted_child.reshape((*extracted_child.shape[1:],))
        t_active_voice_right_child_after_unshift = extract_per_level_tensor_representation_after_unshift(
            extracted_child,
            max_tree_depth=MAX_TREE_DEPTH - 1,
            role_shape=SINGLE_ROLE_SHAPE,
            filler_shape=SINGLE_FILLER_SHAPE)

        for level_idx in range(len(t_V_r0_P_r1) - 1):
            np.testing.assert_array_equal(
                t_V_r0_P_r1[level_idx],
                t_active_voice_right_child_after_unshift[level_idx]
            )

    def test_ideal2(self):
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
        t_active_voice = elementary_join(joiner_network=keras_joiner,
                                         input_structure_max_shape=fillers_shapes,
                                         basic_roles=roles,
                                         basic_fillers=fillers,
                                         subtrees=(
                                             get_filler_by(name='A', order=order_case_active, fillers=fillers),
                                             t_V_r0_P_r1
                                         ))

        dual_basic_roles_case_1 = np.linalg.inv(roles)
        fillers_shapes_unshift = generate_shapes_for_unshift(max_tree_depth=MAX_TREE_DEPTH - 1,
                                                             role_shape=SINGLE_ROLE_SHAPE,
                                                             filler_shape=SINGLE_FILLER_SHAPE)
        keras_ex0_unshifter = build_tree_unshifter_network(roles=dual_basic_roles_case_1,
                                                           fillers_shapes=fillers_shapes_unshift,
                                                           role_index=0)

        prepared_for_unshift = reshape_to_satisfy_max_depth_after_unshift(t_active_voice,
                                                                          MAX_TREE_DEPTH - 1,
                                                                          SINGLE_ROLE_SHAPE,
                                                                          SINGLE_FILLER_SHAPE)

        extracted_child = keras_ex0_unshifter.predict_on_batch([
            *prepared_for_unshift
        ])

        extracted_child = extracted_child.reshape((*extracted_child.shape[1:],))
        t_active_voice_left_child_after_unshift = extract_per_level_tensor_representation_after_unshift(
            extracted_child,
            max_tree_depth=MAX_TREE_DEPTH - 1,
            role_shape=SINGLE_ROLE_SHAPE,
            filler_shape=SINGLE_FILLER_SHAPE)

        # for level_idx in range(len(t_V_r0_P_r1)-1):
        np.testing.assert_array_equal(
            fillers[0],
            t_active_voice_left_child_after_unshift[0][0]
        )
