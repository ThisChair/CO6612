import csv
from random import shuffle

dataset = []
with open('rabbit.csv', newline='') as f:
    r = csv.reader(f)
    for row in r:
        if row:
            dataset.append(row)

shuffle(dataset)
shuffle(dataset)
shuffle(dataset)

train_size = int(len(dataset) * 0.8)

train = dataset[:train_size]
test = dataset[train_size:]
 
with open('rabbit_train.csv', 'w') as f:  
    writer = csv.writer(f)
    writer.writerows(train)

with open('rabbit_test.csv', 'w') as f:  
    writer = csv.writer(f)
    writer.writerows(test)