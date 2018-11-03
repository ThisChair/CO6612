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
ttarget = []
with open('reglin_test.csv', newline='') as f:
    r = csv.reader(f)
    for row in r:
        if row:
            tinput.append([1.0] + [float(row[0])])
            ttarget.append([float(row[1])])
neuronlist = [1, 2, 3, 4, 6, 8, 12, 20, 40]
c = list(zip(input,target))
shuffle(c)
input = [x[0] for x in c]
target = [x[1] for x in c]
input = np.array(input,np.float64)
target = np.array(target,np.float64)
tinput = np.array(tinput,np.float64)
ttarget = np.array(ttarget,np.float64)
for n in neuronlist:
    bp = backpropagation(input,target,n,0.001,700,test=(tinput,ttarget))
    y_list = bp[2]
    x_list = list(range(len(y_list)))
    plt.plot(x_list,y_list)
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.savefig("graphs/"+str(n)+"_neurons_train"+".png",bbox_inches='tight')
    plt.clf()
    y_list = bp[3]
    x_list = list(range(len(y_list)))
    plt.plot(x_list,y_list)
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.savefig("graphs/reglin_"+str(n)+"_neurons_test"+".png",bbox_inches='tight')
    plt.clf()