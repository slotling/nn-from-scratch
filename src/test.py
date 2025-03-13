import model as md
import numpy as np
from dataload import load_csv

train = np.array(load_csv("data/mnist_train.csv", 1))
test = np.array(load_csv("data/mnist_test.csv", 5))

X_train = np.array(np.full((2, 2), 0))
y_train = np.array(np.full((2, 100), 0))
y_train[0][0] = 1

X_train[1][0] = 1
y_train[1][1] = 1

print(X_train, y_train)

models = []

import matplotlib.pyplot as plt

model = md.Model([
    md.LayerDeclaration(X_train.shape[1]),
    md.LayerDeclaration(16),
    md.LayerDeclaration(16),
    md.LayerDeclaration(y_train.shape[1], activation_function="softmax")
])
model.train(X_train, y_train, alpha=2, epoches=1500)
print(model.calculate_and_predict(X_train[0]))
print(model.calculate_and_predict(X_train[1]))

plt.plot(model.debug_cost_list, label=f"alpha = {1}")

plt.legend()
plt.show()