import unittest
import numpy as np

from demo.recursive_structure import pre_process_and_run


class DecodeIntegrationTest(unittest.TestCase):
    def test_ideal1(self):
        """
        First use case for the structure that has nesting equal 1

        root
        | \  \
        A B  C
        """
        # Input information
        fillers_case_1 = np.array([
            [7, 0, 0],  # A
            [0, 13, 0],  # B
            [0, 0, 2],  # C
        ])
        roles_case_1 = np.array([
            [10, 0, 0],  # r_0
            [0, 5, 0],  # r_1
            [0, 0, 8],  # r_2
        ])
        mapping_case_1 = {
            'A': [0],
            'B': [1],
            'C': [2]
        }
        order_case_1 = ['A', 'B', 'C']

        tr_1, fillers_restored_case_1, final_dual_roles_case_1 = pre_process_and_run(roles_vectors=roles_case_1,
                                                                                     fillers_vectors=fillers_case_1,
                                                                                     filler_role_mapping=mapping_case_1,
                                                                                     filler_order=order_case_1)

        fillers_expected = np.array(
            [
                [7., 0., 0.],
                [0., 13., 0.],
                [0., 0., 2.]
            ]
        )
        np.testing.assert_array_equal(fillers_restored_case_1, fillers_expected)

    def test_ideal2(self):
        """
        Second use case for the structure that has variable nesting

        Using this structure for demonstration:
        root
        |  \
        A  / \
          B   \
             / \
            C  D
        """
        # Input information
        # Input information
        fillers_case_2 = np.array([
            [8, 0, 0, 0],  # A
            [0, 15, 0, 0],  # B
            [0, 0, 10, 0],  # C
            [0, 0, 0, 3],  # D
        ])
        roles_case_2 = np.array([
            [10, 0],  # r_0
            [0, 5],  # r_1
        ])
        mapping_case_2 = {
            'A': [0],
            'B': [0, 1],
            'C': [0, 1, 1],
            'D': [1, 1, 1]
        }
        order_case_2 = ['A', 'B', 'C', 'D']

        tr_1, fillers_restored_case_2, final_dual_roles_case_2 = pre_process_and_run(roles_vectors=roles_case_2,
                                                                                     fillers_vectors=fillers_case_2,
                                                                                     filler_role_mapping=mapping_case_2,
                                                                                     filler_order=order_case_2)

        fillers_expected = np.array(
            [
                [8., 0, 0, 0],  # A
                [0, 15., 0, 0],  # B
                [0, 0, 10., 0],  # C
                [0, 0, 0, 3.],  # D
            ]
        )
        np.testing.assert_array_equal(
            np.array(fillers_restored_case_2, dtype=np.int32),
            np.array(fillers_expected, dtype=np.int32)
        )
