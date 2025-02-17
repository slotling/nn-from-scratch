import csv
import numpy as np

def load_csv(path: str, limit: np.int32=-1):
    content = []
    with open(path, mode ='r') as file:
        csvFile = csv.reader(file)

        count = limit
        for i in csvFile:
            array = [int(j) for j in i]
            content.append(array)
            count -= 1
            if count == 0:
                return content
    
    return content

if __name__ == "__main__":
    print(load_csv("data/mnist_train.csv"))