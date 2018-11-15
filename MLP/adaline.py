from math import isclose
from random import random
import numpy as np

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
        last_error = error
        error = 0
        epoch += 1
        for i,o in zip(x,d):
            delta = o - sum([i[k] * w[k] for k in range(inputSize)])
            error += delta ** 2
            w = [w[k] + eta * delta * i[k] for k in range(inputSize)]
        error = error / n
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
