import numpy as np
import csv
import argparse
from random import shuffle

from perceptron import *

parser = argparse.ArgumentParser(
    description='Clasificación de datos usando el algoritmo por reforzamiento.'
)
parser.add_argument(
    '--eta',
    type=float, help='Tasa de aprendizaje. Por defecto es 0.05',
    default=0.05
)
args = parser.parse_args()

input = []
target = []
with open('4D.csv', newline='') as f:
    r = csv.reader(f)
    for row in r:
        if row:
            input.append([1] + row[:-1])
            target.append(row[-1])

c = list(zip(input[1:],target[1:]))
shuffle(c)
input = [x[0] for x in c]
target = [x[1] for x in c]

input = np.array(input,float)
target = np.array(target,int)

w = competitive(input,target,args.eta,15000)

print("Pesos finales: ", w)