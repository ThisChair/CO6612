from math import copysign, isclose
from random import random
import numpy as np

def aprox_zero(x,tolerance):
    return all([isclose(i,0,abs_tol=tolerance) for i in x])

def perceptron(x, d, eta, max_ep, tolerance=1e-09):
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
    while epoch < max_ep:
        epoch += 1
        error = np.array([],float)
        y = np.array([],float)
        for i in range(n):
            y = np.append(y,np.sign(np.dot(np.transpose(w),x[i])))
            error = np.append(error,d[i] - y[i])
            w = w + eta * error[i] * x[i]
        maxErr = max(error)
        meanErr = np.mean(error)
        correct = np.sum(d == y)
        print(
            "Learning Rate: {} Epoch: {} Max Error: {} Mean Error: {} Correct guesses: {} / {}".format(
                eta,
                epoch,
                maxErr,
                meanErr,
                correct,
                n
            )
        )
        if aprox_zero(error,tolerance):
            break
    return w

def competitive(x, d, eta, max_ep):
    '''
    Simple perceptron algorithm.
    
    x: Inputs.
    d: Number of the class each input belongs.
    eta: Learning rate.
    max_ep: Maximum number of epochs allowed.
    '''
    inputSize = x.shape[1]
    c = max(d)
    n = x.shape[0]
    ws = np.array([[random() for _ in range(inputSize)] for _ in range(c)])
    epoch = 0
    while epoch < max_ep:
        epoch += 1
        correct = 0
        for i in range(n):
            y = [np.dot(np.transpose(w),x[i]) for w in ws]
            winner = np.argmax(y) + 1
            if winner != d[i]:
                ws[d[i]-1] = ws[d[i]-1] + eta * x[i]
                ws[winner-1] = ws[winner-1] - eta * x[i]
            else:
                correct += 1
        print(
            "Learning rate: {} Epoch: {} Correct guesses: {} / {}".format(
                eta,
                epoch,
                correct,
                n
            )
        )
        if correct == n:
            break
    return ws
