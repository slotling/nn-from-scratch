import model as md
import numpy as np
from dataload import load_csv

train = np.array(load_csv("data/mnist_train.csv", 1))
test = np.array(load_csv("data/mnist_test.csv", 5))

X_train = np.array([np.full((2), 0)])
y_train = np.array([np.full((2), 1)])
y_train[0][0] = 0

models = []

import matplotlib.pyplot as plt
for alpha in [0.1, 1]:
    model = md.Model([X_train.shape[1], 64, y_train.shape[1]])
    model.train(X_train, y_train, alpha=alpha)
    print(model.predict(X_train[0]))

    plt.plot(model.debug_cost_list, label=f"alpha = {alpha}")

plt.legend()
plt.show()