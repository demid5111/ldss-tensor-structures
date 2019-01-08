import numpy as np

from src.layers.ilayer import ILayer


class BindingCell(ILayer):
    """
    First input is fillers matrix and second is roles matrix
    """

    def forward(self, inputs, **kwargs):
        """
        Performs tensor multiplication (matrix by matrix)
        :param inputs: input matrix
        :param kwargs: not used
        """
        if len(inputs) != 2:
            raise ValueError('Binding Cell accepts only two inputs')
        fillers = inputs[0]
        roles = inputs[1]
        if len(fillers) != len(roles):
            raise ValueError('Binding Cell accepts as inputs two matrixes of equal number of rows')
        res = [np.outer(fillers[i], roles[i]) for i in range(len(fillers))]
        self.output = np.array(res)
