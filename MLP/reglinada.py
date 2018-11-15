import numpy as np
import csv
import argparse
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt

from adaline import *

parser = argparse.ArgumentParser(
    description='Clasificación de datos rabit.'
)
parser.add_argument(
    '--eta',
    type=float,
    help='Tasa de aprendizaje. Por defecto es 0.001',
    default=0.001
)

parser.add_argument(
    '--n',
    type=int,
    help='Grado del polinomio. 1 por defecto',
    default=1
)

args = parser.parse_args()

input = []
target = []
with open('reglin_train.csv', newline='') as f:
    r = csv.reader(f)
    for row in r:
        if row:
            x = float(row[0])
            input.append([1.0] + [x**(i+1) for i in range(args.n)])
            target.append([float(row[1])])

tinput = []
ttarget = []
with open('reglin_test.csv', newline='') as f:
    r = csv.reader(f)
    for row in r:
        if row:
            x = float(row[0])
            tinput.append([1.0] + [x**(i+1) for i in range(args.n)])
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

w = lms(input,target,args.eta,700)
x_list = [x[1] for x in tinput]
func = [sum([x[i] * w[i] for i in range(len(w))]) for x in tinput]
y_list = [y[0] for y in ttarget]
terror = sum([(y_list[i] - func[i]) ** 2 for i in range(len(func))]) / len(func)
print("Training error: ",terror)
plt.plot(x_list,func, label='Función entrenada',c='r')
plt.scatter(x_list,y_list,label='Conjunto de prueba', c='g', marker='.')
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc='upper left')
plt.savefig("graphs/reglin_"+str(args.n)+"poly_function"+".png",bbox_inches='tight')
plt.clf()