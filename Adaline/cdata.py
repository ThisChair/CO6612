import numpy as np
import csv
import argparse
from random import shuffle
import matplotlib.pyplot as plt

from perceptron import *
from adaline import *

parser = argparse.ArgumentParser(
    description='Clasificación usando LMS con función de activación logística.'
)
parser.add_argument(
    '--eta',
    type=float, help='Tasa de aprendizaje. Por defecto es 0.05',
    default=0.05
)
args = parser.parse_args()

input = []
target = []
cat1 = []
cat2 = []
with open('classdata.csv', newline='') as f:
    r = csv.reader(f)
    for row in r:
        if row:
            input.append([1] + row[:-1])
            if row[-1] == '0': 
                cat1.append(row[:-1])
            else:
                cat2.append(row[:-1])
            target.append(row[-1])

c = list(zip(input,target))
shuffle(c)
input = [x[0] for x in c]
target = [x[1] for x in c]



input = np.array(input,float)
target = np.array(target,float)

targetp = np.array([x * 2 - 1 for x in target])

w = lms(input,target,args.eta,100000)
wp = perceptron(input,targetp,args.eta,100000)

validation = []
with open('classdataVal.csv', newline='') as f:
    r = csv.reader(f)
    for row in r:
        if row:
            validation.append(row)

validation = np.transpose(np.array(validation,float))
cat1 = np.transpose(np.array(cat1,float))
cat2 = np.transpose(np.array(cat2,float))
plt.scatter(cat1[0],cat1[1], c="b", marker=".", label="Clase 1 (Entrenamiento)")
plt.scatter(cat2[0],cat2[1], c="r", marker=".", label="Clase 2 (Entrenamiento)")
f = lambda x: (-x * w[1]- w[0])/w[2]
r = np.arange(-1.0,1.1,0.1)
p = list(map(f,r))
plt.plot(r,p, label="LMS Logística", c="g")
f = lambda x: (-x * wp[1]- wp[0])/wp[2]
p = list(map(f,r))
plt.plot(r,p, label="Perceptron", c="y")
plt.scatter(validation[0],validation[1], c="purple", marker="^", label="Datos de validación")
plt.axis([-1,1,-1,1])
plt.legend(loc='upper left')
plt.show()