import unittest

import numpy as np
import keras.backend as K

from core.decoder.vendor.network import build_filler_decoder_network


class DecoderNetworkTest(unittest.TestCase):
    def test_decoder_ideal1(self):
        """
        First use case for the structure that has nesting equal 1

        root
        | \  \
        A B  C
        """
        encoded_structure_blob = np.array(
            [
                [
                    [70., 0., 0.],
                    [0., 65., 0.],
                    [0., 0., 16.]
                ]
            ]
        )

        dual_roles_shape = (1, 3, 3)
        final_dual_roles = np.array(
            [
                [0.1, 0., 0.],
                [0., 0.2, 0.],
                [0., 0., 0.125]
            ]
        )
        reshaped_dual_roles = final_dual_roles.reshape(dual_roles_shape)

        tr_shape = (1, 3, 3)
        keras_decoder = build_filler_decoder_network(input_shapes=(tr_shape, dual_roles_shape))
        # with K.get_session():
        fillers_restored = keras_decoder.predict_on_batch([
            encoded_structure_blob,
            reshaped_dual_roles
        ])

        fillers_expected = np.array(
            [
                [7., 0., 0.],
                [0., 13., 0.],
                [0., 0., 2.]
            ]
        )
        np.testing.assert_array_equal(fillers_restored, fillers_expected)

    def test_decoder_ideal2(self):
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
        encoded_structure_blob = np.array(
            [
                [
                    [80., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 750., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 2500., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 375.]
                ]
            ]
        )

        dual_roles_shape = (1, 4, 8)
        final_dual_roles = np.array(
            [
                [0.1, 0., 0., 0., 0., 0., 0., 0.],
                [0., 0.02, 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0.004, 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0.008]
            ]
        )
        reshaped_dual_roles = final_dual_roles.reshape(dual_roles_shape)

        tr_shape = (1, 4, 8)
        keras_decoder = build_filler_decoder_network(input_shapes=(tr_shape, dual_roles_shape))

        # with K.get_session():
        fillers_restored = keras_decoder.predict_on_batch([
            encoded_structure_blob,
            reshaped_dual_roles
        ])

        fillers_expected = np.array(
            [
                [8., 0., 0., 0.],
                [0., 15., 0., 0.],
                [0., 0., 10., 0.],
                [0., 0., 0., 3.0000002]
            ]
        )
        np.testing.assert_array_equal(
            np.array(fillers_restored, dtype=np.int32),
            np.array(fillers_expected, dtype=np.int32)
        )
