import numpy as np
import csv
import argparse
from random import shuffle
from math import sin,pi
import numpy as np
import matplotlib.pyplot as plt

from rbf import RBF


xs = np.sort(np.random.uniform(0,1,100))

h = lambda x: 0.5 + 0.4 * sin(2*pi*x)
h = np.vectorize(h, otypes=[np.float])

ys = h(xs)

noise = np.random.normal(0, 1, ys.shape)
ys_noise = ys + noise

input = [[x] for x in xs]
shuffle(input)
input = np.array(input,np.float64)

ns = [10,20,25,50,60,70,80]
sigmas = [0.1,0.25,0.5,0.75,1,100,200,300,400,500]
regs = [0,1,0.5,0.1,0.01,0.001]
for n in ns:
    for sigma in sigmas:
        for reg in regs:
            net = RBF(len(input),n,sigma)
            net.train(input,ys_noise,reg)
            approx = [net.predict(x) for x in xs]
            plt.plot(xs,ys,label='Función',c='b')
            plt.plot(xs,ys_noise,label='Función con ruido',c='g')
            plt.plot(xs,approx,label='Aproximación',c='r')
            plt.legend(loc='upper left')
            error = sum([(ys_noise[i] - approx[i]) ** 2 for  i in range(len(ys_noise))]) / len(ys_noise)
            print("Centros: {} | Sigma: {} | Lambda: {} | Error: {}".format(
                n,
                sigma,
                reg,
                error
            ))
            plt.savefig(
                "graphs/centers_"+str(n)+"_sigma_"+str(sigma)+"_lambda_"+str(reg)+".png",
                bbox_inches='tight'
            )
            plt.clf()