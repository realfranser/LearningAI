"""
Neural Network Chapter 1
Python 2.7
"""

import numpy as np

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

class Network(object):

    def __init__(self, sizes):
        """
        I.E. -> NN with 2 neurons in the first layer, 3 in the second, and 1 in
        the final:
        net = Network([2, 3, 1])
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1],
            sizes[1:])]

