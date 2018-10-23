from random import random
import matplotlib.pyplot as plt
import numpy as np

from perceptron import aprox_zero

def gradient_descent(w,eta):
    epoch = 0
    ws = [[x for x in w]]
    while True:
        epoch += 1
        u0 = w[0] + 0.8182 * w[1] - 0.8182
        u1 = w[1] + 0.8182 * w[0] - 0.354
        print(
            "Learning Rate: {} Epoch: {} Gradient: {} W: {}".format(
                eta,
                epoch,
                [u0,u1],
                w
            )
        )
        if (aprox_zero([u0,u1],1e-09)):
            break
        w[0] = w[0] - eta * u0
        w[1] = w[1] - eta * u1
        ws.append([x for x in w])
    return ws

w = [random(),random()]

gradient1 = gradient_descent([x for x in w],0.3)
gradient2 = gradient_descent([x for x in w],1.0)

print("Result with 0.3: ",gradient1[-1])
print("Result with 1: ",gradient2[-1])

gradient1 = np.transpose(gradient1)
gradient2 = np.transpose(gradient2)

plt.plot(gradient1[0],gradient1[1],label="LR = 0.3", c="b")
plt.plot(gradient2[0],gradient2[1],label="LR = 1", c="r")
#plt.axis([-1,1,-1,1])
plt.legend(loc='upper left')
plt.show()