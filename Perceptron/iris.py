import numpy as np
import csv
import argparse
from random import shuffle

from perceptron import *

parser = argparse.ArgumentParser(
    description='Clasificación de Iris setosa usando el algoritmo del perceptron.'
)
parser.add_argument(
    '--eta',
    type=float, help='Tasa de aprendizaje. Por defecto es 0.05',
    default=0.05
)
args = parser.parse_args()

input = []
target = []
with open('bezdekIris.data', newline='') as f:
    r = csv.reader(f)
    for row in r:
        if row:
            input.append([1] + row[:-1])
            if row[-1] == 'Iris-setosa':
                target.append(1)
            else:
                target.append(-1)

c = list(zip(input,target))
shuffle(c)
input = [x[0] for x in c]
target = [x[1] for x in c]

input = np.array(input,float)
target = np.array(target)

perceptron(input,target,args.eta,4000)