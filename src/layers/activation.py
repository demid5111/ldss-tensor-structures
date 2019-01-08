import numpy as np

from src.layers.ilayer import ILayer


class Activation(ILayer):
    """
    Performs activation over corresponding tensor
    """
    def __init__(self, t):
        super().__init__()
        self.type = t

    def forward(self, inputs, **kwargs):
        """
        Performs specified activation operation over a single input
        :param inputs: input matrix
        :param kwargs: not used
        """
        if len(inputs) != 1:
            raise ValueError('Activation layer accepts only one tensor as input')

        if self.type != 'ReLU':
            raise NotImplementedError('Activation layer accepts only ReLU')

        self.output = inputs[0] * (inputs[0] > 0)
