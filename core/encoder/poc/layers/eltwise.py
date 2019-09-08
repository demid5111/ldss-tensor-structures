import numpy as np

from core.encoder.poc.layers.ilayer import ILayer


class Eltwise(ILayer):
    """
    Takes x inputs of equal size and performs operation over corresponding items
    """
    def __init__(self, t):
        super().__init__()
        self.type = t

    def forward(self, inputs, **kwargs):
        """
        Performs element-wise operation over a set of matrices
        :param inputs: input matrix
        :param kwargs: not used
        """
        all_same_shapes = [i.shape == inputs[0].shape for i in inputs]
        if not all(all_same_shapes):
            raise ValueError('Element-wise layer accepts tensors of equal shape')

        if self.type != 'sum':
            raise NotImplementedError('Element-wise layer accepts only summation')

        self.output = np.sum(inputs[0], axis=0)