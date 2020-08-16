import numpy as np


class SGD(object):
    def __init__(self, model, learning_rate, momentum=0.0):
        self.model = model
        self.learning_rate = learning_rate
        self.momentum = momentum

    def step(self):
        """One backpropagation step, update weights layer by layer"""
        layers = self.model.layerList

        for layer in layers:
            if layer.trainable:

                ############################################################################
                # TODO: Put your code here
                # Calculate diff_W and diff_b using layer.grad_W and layer.grad_b.
                # You need to add momentum to this.

                # Weight update without momentum
                layer.v_W = self.momentum * layer.v_W + layer.grad_W
                layer.W += -self.learning_rate*layer.v_W

                layer.v_b = self.momentum * layer.v_b + layer.grad_b
                layer.b += -self.learning_rate * layer.v_b

                ############################################################################
