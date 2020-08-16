import numpy as np


class ReLULayer(object):

    def __init__(self):
        """
        Applies the rectified linear unit function element-wise: relu(x) = max(x, 0)
        """

        self.trainable = False # no parameters

    def forward(self, Input):

        ############################################################################
        # TODO: Put your code here
        # Apply ReLU activation function to Input, and return results.
        self.Input = Input
        out = np.ones(Input.shape)
        out[(Input < 0 )] = 0
        return out
        ############################################################################




    def backward(self, delta):

        ############################################################################
        # TODO: Put your code here
        # Calculate the gradient using the later layer's local sensitivity: delta
        y = self.Input
        y[y>0] = 1
        y[y<0] = 0
        delta = np.multiply(delta, y)
        return delta
        ############################################################################

