import numpy as np

# a small number to prevent dividing by zero, maybe useful for you
EPS = 1e-11

class SoftmaxCrossEntropyLoss(object):

    def forward(self, logits, labels):
        """
          Inputs: (minibatch)
          - logits: forward results from the last FCLayer, shape (batch_size, 10)
          - labels: the ground truth label, shape (batch_size, )
        """

        ############################################################################
        # TODO: Put your code here
        # Calculate the average accuracy and loss over the minibatch
        # Return the loss and acc, which will be used in solver.py
        # Hint: Maybe you need to save some arrays for backward

        self.one_hot_labels = np.zeros_like(logits)
        self.one_hot_labels[np.arange(len(logits)), labels] = 1

        self.prob = np.exp(logits) / (EPS + np.exp(logits).sum(axis=1, keepdims=True))

        # calculate the accuracy
        preds = np.argmax(self.prob, axis=1)  # self.prob, not logits.
        acc = np.mean(preds == labels)

        # calculate the loss
        loss = np.sum(-self.one_hot_labels * np.log(self.prob + EPS), axis=1)
        loss = np.mean(loss)

        ############################################################################


        return loss, acc

    def backward(self):

        ############################################################################
        # TODO: Put your code here
        # Calculate and return the gradient (have the same shape as logits)
        
        return self.prob - self.one_hot_labels
        ############################################################################
