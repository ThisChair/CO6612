from random import random
from statistics import mean,stdev
import numpy as np

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def feedforward(x, hw, ow):
    hy = [sigmoid(sum([hw[i][j] * x[j] for j in range(len(x))])) for i in range(len(hw))]
    hy = np.array([1] + hy,np.float64)
    oy = [sum([ow[i][j] * hy[j] for j in range(len(hy))]) for i in range(len(ow))]
    oy = np.array(oy,np.float64)
    return hy,oy

def standardize(x):
    size = len(x)
    new_x = np.transpose(x)
    features = len(new_x)
    means = [mean(z) for z in new_x]
    stdevs = [stdev(new_x[i],means[i]) for i in range(features)]
    res = []
    for j in range(size):
        res.append([])
        for i in range(features):
            if stdevs[i] == 0:
                res[j].append(new_x[i][j])
            else:
                res[j].append(
                    (new_x[i][j] - means[i])/stdevs[i]
                )
    res = np.array(res,np.float64)
    return res

def backpropagation(xs, d, nh, eta, maxepoch,premature_stop=False,test=None):
    ni = len(xs[0])
    no = len(d[0])
    hw = np.array([[random() for _ in range(ni)] for _ in range(nh)],np.float64)
    ow = np.array([[random() for _ in range(nh+1)] for _ in range(no)],np.float64)
    epoch = 0
    errors = []
    test_errors = []
    previous = 0
    while epoch < maxepoch:
        epoch += 1
        mse = 0
        for (x,t) in zip(xs,d):
            hy,oy = feedforward(x, hw, ow)
            do = np.array([t[k]-oy[k] for k in range(no)],np.float64)
            hy = hy
            sums = np.array([sum([ow[j][k] * do[j] for j in range(len(do))]) for k in range(nh)],np.float64)
            dh = [hy[1:][k] * (1 - hy[1:][k]) * sums[k] for k in range(nh)]
            ow = [[ow[j][i] + eta * do[j] * hy[i] for i in range(nh+1)] for j in range(no)]
            hw = [[hw[j][i] + eta * dh[j] * x[i] for i in range(ni)] for j in range(nh)]

            mse += sum([(t[k]-oy[k])**2 for k in range(len(oy))])
        mse = mse/len(xs)
        print(
            "Learning Rate: {} Epoch: {} Error: {}".format(
                eta,
                epoch,
                mse
            )
        )
        errors.append(mse)
        if test:
            ys = [feedforward(x, hw, ow)[1] for x in test[0]]
            mset = sum([sum([(test[1][j][k]-ys[j][k])**2 for k in range(len(oy))]) for j in range(len(test[1]))])
            mset = mset/len(ys)
            test_errors.append(mset)
            print("Test errors: ",mset)
        if premature_stop and (mse - previous < 1e-09):
            break
        previous = mse
    return hw, ow, errors, test_errors

def predict(x, hw, ow):
    hy,oy = feedforward(x, hw, ow)
    return oy