import unittest
import numpy as np

from core.tensor_math.math_utils import matrix_multiplication


class MatrixMultiplicationTest(unittest.TestCase):
    @staticmethod
    def test_matrix_tensor():
        m = np.array([
            [2, 3],
            [1, 5]
        ], dtype=np.int64)
        t = np.array([10, 20], dtype=np.int64)
        res = matrix_multiplication(m, t)
        # as it is tensor multiplication, dimensions a summarized
        # matrix dim = 2
        # vector dim = 1
        # (matrix * vector) dim=3
        exp_res = np.array([
            [
                [20, 40],
                [30, 60]
            ],
            [
                [10, 20],
                [50, 100]
            ],
        ])
        np.testing.assert_array_equal(res, exp_res)

    @staticmethod
    def test_matrix_matrix():
        m1 = np.array([
            [2, 3],
            [1, 5]
        ], dtype=np.int64)
        m2 = np.array([
            [7, 8],
            [9, 10]
        ], dtype=np.int64)
        res = matrix_multiplication(m1, m2)
        # as it is tensor multiplication, dimensions a summarized
        # matrix1 dim = 2
        # matrix2 dim = 2
        # vector dim = 1
        # (matrix1 * matrix2) dim=4
        exp_res = np.array([
            [
                [
                    [14, 16],
                    [18, 20]
                ],
                [
                    [21, 24],
                    [27, 30]
                ]
            ],
            [
                [
                    [7, 8],
                    [9, 10]
                ],
                [
                    [35, 40],
                    [45, 50]
                ]
            ]
        ])
        np.testing.assert_array_equal(res, exp_res)
