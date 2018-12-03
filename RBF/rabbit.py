import numpy as np
import csv
import argparse
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt

from rbf import RBF

parser = argparse.ArgumentParser(
    description='Interpolación de datos rabbit. Usando RBF'
)
parser.add_argument(
    '--n',
    type=int,
    help='Número de centros. Por defecto son 5.',
    default=5
)

parser.add_argument(
    '--sigma',
    type=float,
    help='Parámetro sigma. Por defecto es 1',
    default=1.0
)

parser.add_argument(
    '--reg',
    type=float,
    help='Parámetro lambda de regularización. Por defecto es 0',
    default=0.0
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
            oinput.append([float(row[0])])
            otarget.append(float(row[1]))

with open('rabbit_train.csv', newline='') as f:
    r = csv.reader(f)
    for row in r:
        if row:
            input.append([float(row[0])])
            target.append(float(row[1]))

with open('rabbit_test.csv', newline='') as f:
    r = csv.reader(f)
    for row in r:
        if row:
            test.append(row)


c = list(zip(input,target))
shuffle(c)
input = [x[0] for x in c]
target = [x[1] for x in c]
input = np.array(input,np.float64)
target = np.array(target,np.float64)
test = np.transpose(np.array(test,np.float64))
print(len(input))
net = RBF(len(input),args.n,args.sigma)
net.train(input,target,args.reg)
x_list = [x[0] for x in oinput]
y_list = [net.predict(x) for x in oinput]
yo_list = [x for x in otarget]
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