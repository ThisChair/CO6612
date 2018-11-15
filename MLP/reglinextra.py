import numpy as np
import csv
import argparse
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt

from backpropagation import *

input = []
target = []
with open('reglin_train.csv', newline='') as f:
    r = csv.reader(f)
    for row in r:
        if row:
            input.append([1.0] + [float(row[0])])
            target.append([float(row[1])])

tinput = []
x_list = []
ttarget = []
with open('reglin_test.csv', newline='') as f:
    r = csv.reader(f)
    for row in r:
        if row:
            tinput.append([1.0] + [float(row[0])])
            ttarget.append([float(row[1])])
neuronlist = [2, 8, 40]
c = list(zip(input,target))
shuffle(c)
input = [x[0] for x in c]
target = [x[1] for x in c]
input = np.array(input,np.float64)
target = np.array(target,np.float64)
tinput = np.array(tinput,np.float64)
ttarget = np.array(ttarget,np.float64)
for n in neuronlist:
    bp = backpropagation(input,target,n,0.001,700)
    x_list = [x[1] for x in tinput]
    func = [predict(x,bp[0],bp[1])[0] for x in tinput]
    y_list = [y[0] for y in ttarget]
    plt.plot(x_list,func, label='Función entrenada',c='r')
    plt.scatter(x_list,y_list,label='Conjunto de prueba', c='g', marker='.')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc='upper left')
    plt.savefig("graphs/reglin_"+str(n)+"_neurons_function"+".png",bbox_inches='tight')
    plt.clf()