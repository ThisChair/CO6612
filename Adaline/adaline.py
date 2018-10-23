from math import isclose
from random import random
import numpy as np

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def lms(x, d, eta, max_ep, tolerance=1e-09):
    '''
    Simple perceptron algorithm.
    
    x: Inputs.
    d: 1 if x in a class, either -1.
    eta: Learning rate.
    max_ep: Maximum number of epochs allowed.
    tolerance: Absolute tolerance to say that a number approximates to zero.
    '''
    inputSize = x.shape[1]
    n = x.shape[0]
    w = np.array([random() for _ in range(inputSize)])
    epoch = 0
    error = 0
    while epoch < max_ep:
        epoch += 1
        y = sigmoid(np.dot(x,w))
        last_error = error
        error = 0.5 * sum([d[k] - y[k] for k in range(len(x))])
        delta = np.array(
            [np.sum([(d[k] - y[k]) * y[k] * (1 - y[k]) * x[k][i] for k in range(len(x))]) for i in range(len(w))]
        )
        w = w + eta * delta
        
        print(
            "Learning Rate: {} Epoch: {} Error: {} Difference: {}".format(
                eta,
                epoch,
                error,
                last_error - error
            )
        )
        
        if isclose(error,last_error):
            break
    return w
