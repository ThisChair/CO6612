import numpy as np
import csv
import argparse
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt

from backpropagation import *

parser = argparse.ArgumentParser(
    description='Clasificación de datos rabit.'
)
parser.add_argument(
    '--eta',
    type=float,
    help='Tasa de aprendizaje. Por defecto es 0.05',
    default=0.05
)

parser.add_argument(
    '--n',
    type=int,
    help='Número de neuronas en capa oculta. Por defecto es 2.',
    default=2
)

parser.add_argument(
    '--epochs',
    type=int,
    help='Número de épocas durante las cuales entrenar. Por defecto es 400.',
    default=400
)

args = parser.parse_args()

input = []
target = []
test = []
oinput = []
otarget = []

with open('rabbit.csv', newline='') as f:
    r = csv.reader(f)
    for row in r:
        if row:
            oinput.append([1.0] + [float(row[0])])
            otarget.append([float(row[1])])

with open('rabbit_train.csv', newline='') as f:
    r = csv.reader(f)
    for row in r:
        if row:
            input.append([1.0] + [float(row[0])])
            target.append([float(row[1])])

with open('rabbit_test.csv', newline='') as f:
    r = csv.reader(f)
    for row in r:
        if row:
            test.append(row)


neuronlist = [1, 2, 3, 4, 6, 8, 12, 20, 40]
c = list(zip(input,target))
shuffle(c)
input = [x[0] for x in c]
target = [x[1] for x in c]
input = np.array(input,np.float64)
input = standardize(input)
target = np.array(target,np.float64)
test = np.transpose(np.array(test,np.float64))

bp = backpropagation(input,target,args.n,args.eta,args.epochs)
x_list = [x[1] for x in oinput]
y_list = [predict(x,bp[0],bp[1])[0] for x in standardize(oinput)]
yo_list = [x[0] for x in otarget]
f = lambda x: 233.846 * (1-np.exp(-0.00604 * x))
yf_list = [f(x) for x in x_list]
error = sum([(y_list[i] - yo_list[i]) ** 2 for  i in range(len(y_list))]) / len(y_list)
print("Error: ",error)
plt.plot(x_list,y_list,label='Predicción',c='b')
plt.plot(x_list,yo_list,label='Datos', c='r')
plt.plot(x_list,yf_list,label='Modelo proporcionado', c='y')
plt.scatter(test[0],test[1],label='Conjunto de prueba', c='g', marker='.')
plt.xlabel("Edad")
plt.ylabel("Peso")
plt.legend(loc='upper left')
#plt.savefig("graphs/rabbit_"+str(args.n)+"_neurons_"+str(args.eta)+".png",bbox_inches='tight')
plt.show()
plt.clf()