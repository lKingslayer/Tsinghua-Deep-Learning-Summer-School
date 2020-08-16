import numpy as np


class FCLayer(object):

    def __init__(self, num_input, num_output, act_function='relu', trainable=True):
        """
        Apply a linear transformation to the incoming data: y = Wx + b
        Args:
            num_input: size of each input sample
            num_output: size of each output sample
            act_function: the name of the activation function such as 'relu', 'sigmoid'
            trainable: whether if this layer is trainable
        """

        self.num_input = num_input
        self.num_output = num_output
        self.trainable = trainable
        self.act_function = act_function
        assert act_function in ['relu', 'sigmoid']

        self.XavierInit()

    def forward(self, Input):

        ############################################################################
        # TODO: Put your code here
        # Apply linear transformation(Wx+b) to Input, and return results.
        self.Input = Input
        output = np.dot(Input, self.W) + self.b
        return output
        ############################################################################


    def backward(self, delta):
        ############################################################################
        # TODO: Put your code here
        # Calculate the gradient using the later layer's gradient: delta

        # self.grad_W and self.grad_b will be used in optimizer.py
        batch_size = np.double(np.shape(self.Input)[0])
        self.grad_b = np.mean(delta,0)
        self.grad_W = np.dot(np.transpose(self.Input),delta)/batch_size
        delta = np.dot(delta, self.W.T)
        return delta
        ############################################################################





    def XavierInit(self):
        """
        Initialize the weigths according to the type of activation function.
        """

        raw_std = (2 / (self.num_input + self.num_output))**0.5
        if 'relu' == self.act_function:
            init_std = raw_std * (2**0.5)
        elif 'sigmoid' == self.act_function:
            init_std = raw_std
        else:
            init_std = raw_std # * 4

        self.W = np.random.normal(0, init_std, (self.num_input, self.num_output))
        self.b = np.random.normal(0, init_std, (1, self.num_output))
        self.v_W = 0
        self.v_b = 0
